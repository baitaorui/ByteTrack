import numpy as np  
from .basetrack import TrackState, STrack, BaseTracker
from .reid_models.deepsort_reid import Extractor
from . import matching
import torch 
from torchvision.ops import nms
from .kalman_filter import *

class NewByteTrack(BaseTracker):
    def __init__(self, opts, frame_rate=30, gamma=0.1, *args, **kwargs) -> None:
        super().__init__(opts, frame_rate, *args, **kwargs)
        self.use_apperance_model = False
        # self.reid_model = Extractor(opts.reid_model_path, use_cuda=True)
        self.gamma = gamma  # coef that balance the apperance and ious

        self.low_conf_thresh = max(0.15, self.opts.match_thresh - 0.3)  # low threshold for second matching

        self.filter_small_area = False  # filter area < 50 bboxs
        self.kalman = NewKalmanFilter()
        self.kalman_format = 'new'
        # self.kalman = KalmanFilter()
        # self.kalman_format = 'default'

    def get_feature(self, tlbrs, ori_img):
        """
        get apperance feature of an object
        tlbrs: shape (num_of_objects, 4)
        ori_img: original image, np.ndarray, shape(H, W, C)
        """
        obj_bbox = []

        for tlbr in tlbrs:
            tlbr = list(map(int, tlbr))

            obj_bbox.append(
                ori_img[tlbr[1]: tlbr[3], tlbr[0]: tlbr[2]]
            )
        
        if obj_bbox:  # obj_bbox is not []
            features = self.reid_model(obj_bbox)  # shape: (num_of_objects, feature_dim)
        else:
            features = np.array([])

        return features

    def update2(self, det_results, img_info, ori_img):
        """
        this func is called by every time step

        det_results: numpy.ndarray or torch.Tensor, shape(N, 6), 6 includes bbox, conf_score, cls
        ori_img: original image, np.ndarray, shape(H, W, C)
        """

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        scores = det_results[:, 4]
        bboxes = det_results[:, :4]

        """step 1. filter results and init tracks"""

        # cal high and low indicies
        det_high_indicies = det_results[:, 4] >= self.det_thresh
        det_low_indicies = np.logical_and(np.logical_not(det_high_indicies), det_results[:, 4] > self.low_conf_thresh)

        # init saperatly
        det_high, det_low = det_results[det_high_indicies], det_results[det_low_indicies]
        bboxes_high = bboxes[det_high_indicies]
        bboxes_low = bboxes[det_low_indicies]
        scores_high = scores[det_high_indicies]
        scores_low = scores[det_low_indicies]

        def tlbr_to_tlwh(tlbr):
            ret = np.asarray(tlbr).copy()
            ret[2:] -= ret[:2]
            return ret

        if det_high.shape[0] > 0:
            D_high = [STrack(cls, tlbr_to_tlwh(xyxy), score, kalman_format=self.kalman_format)
                            for (cls, xyxy, score) in zip([0 for _ in bboxes_high], bboxes_high, scores_high)]
        else:
            D_high = []

        if det_low.shape[0] > 0:
            D_low = [STrack(cls, tlbr_to_tlwh(xyxy), score, kalman_format=self.kalman_format)
                            for (cls, xyxy, score) in zip([0 for _ in bboxes_low], bboxes_low, scores_low)]
        else:
            D_low = []

        # Do some updates
        unconfirmed = []  # unconfirmed means when frame id > 2, new track of last frame
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
       
        # update track state
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Kalman predict, update every mean and cov of tracks
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)

        """Step 2. first match, match high conf det with tracks"""
        if self.use_apperance_model:
            # use apperance model, DeepSORT way
            Apperance_dist = matching.embedding_distance(strack_pool, D_high, metric='cosine')
            IoU_dist = matching.iou_distance(atracks=strack_pool, btracks=D_high)
            Dist_mat = self.gamma * IoU_dist + (1. - self.gamma) * Apperance_dist
        else:
            Dist_mat = matching.iou_distance(atracks=strack_pool, btracks=D_high)
        
        # match
        matched_pair0, u_tracks0_idx, u_dets0_idx = matching.linear_assignment(Dist_mat, thresh=0.9)
        for itrack_match, idet_match in matched_pair0:
            track = strack_pool[itrack_match]
            det = D_high[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)

        u_tracks0 = [strack_pool[i] for i in u_tracks0_idx if strack_pool[i].state == TrackState.Tracked]
        u_dets0 = [D_high[i] for i in u_dets0_idx]

        """Step 3. second match, match remain tracks and low conf dets"""
        # only IoU
        Dist_mat = matching.iou_distance(atracks=u_tracks0, btracks=D_low)
        matched_pair1, u_tracks1_idx, u_dets1_idx = matching.linear_assignment(Dist_mat, thresh=0.5)

        for itrack_match, idet_match in matched_pair1:
            track = u_tracks0[itrack_match]
            det = D_low[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)
        
        """ Step 4. deal with rest tracks and dets"""
        # # deal with final unmatched tracks
        for idx in u_tracks1_idx:
            track = strack_pool[idx]
            track.mark_lost()
            lost_stracks.append(track)
        
        # deal with unconfirmed tracks, match new track of last frame and new high conf det
        Dist_mat = matching.iou_distance(unconfirmed, u_dets0)
        matched_pair2, u_tracks2_idx, u_dets2_idx = matching.linear_assignment(Dist_mat, thresh=0.7)
        for itrack_match, idet_match in matched_pair2:
            track = unconfirmed[itrack_match]
            det = u_dets0[idet_match]
            track.update(det, self.frame_id)
            activated_starcks.append(track)
            
        # deal with lost stracks and new det
        matched_new_det = []
        # if len(u_dets2_idx) != 0:
        #     lost_tracks = [lst for lst in strack_pool if (lst.state == TrackState.Lost and self.frame_id - lst.end_frame <= self.max_time_lost)]
        #     new_det = [u_dets0[i] for i in u_dets2_idx if u_dets0[i].score > self.det_thresh]
        #     # for t in lost_tracks:
        #     #     min = 1000000
        #     #     inx = None
        #     #     ii = -1
        #     #     for i in range(len(new_det)):
        #     #         d = new_det[i]
        #     #         cosine = np.sqrt(np.sum(np.square(t.tlwh[:2] - d.tlwh[:2])))
        #     #         if cosine < min:
        #     #             inx = d
        #     #             min = cosine
        #     #             ii = i
        #     #     if inx is not None:
        #     #         matched_new_det.append(inx)
        #     #         t.re_activate(inx, self.frame_id)
        #     #         refind_stracks.append(t)
        #     #         del new_det[ii]
        #     dis = matching.euc_distance(lost_tracks, new_det)
        #     # mp, ut, ud = matching.linear_assignment(dis, thresh=170)
        #     # for i, j in mp:
        #     #     t = lost_tracks[i]
        #     #     d = new_det[j]
        #     #     matched_new_det.append(d)
        #     #     t.re_activate(d, self.frame_id)
        #     #     refind_stracks.append(t)
        #     #     print("bbbbbbbb")
        #     for t in range(len(lost_tracks)):
        #         inx = None
        #         ii = -1
        #         tr = lost_tracks[t]
        #         for i in range(len(new_det)):
        #             d = new_det[i]
        #             cosine = dis[t][i]
        #             thr = (self.frame_id - tr.frame_id) * 10.8
        #             if cosine < thr:
        #                 inx = d
        #                 ii = i
        #                 break
        #         if inx is not None:
        #             matched_new_det.append(inx)
        #             t.re_activate(inx, self.frame_id)
        #             refind_stracks.append(t)
        #             del new_det[ii]
            
            
        
        # # deal with final unmatched tracks
        # for idx in u_tracks1_idx:
        #     track = strack_pool[idx]
        #     track.mark_lost()
        #     lost_stracks.append(track)

        for idx in u_tracks2_idx:
            track = unconfirmed[idx]
            track.mark_removed()
            removed_stracks.append(track)

        # deal with new tracks
        for idx in u_dets2_idx:
            det = u_dets0[idx]
            if det not in matched_new_det and det.score > self.det_thresh + 0.1:
                det.activate(self.frame_id)
                activated_starcks.append(det)

        """ Step 5. remove long lost tracks"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # update all
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)


        # print
        if self.debug_mode:
            print('===========Frame {}=========='.format(self.frame_id))
            print('Activated: {}'.format([track.track_id for track in activated_starcks]))
            print('Refind: {}'.format([track.track_id for track in refind_stracks]))
            print('Lost: {}'.format([track.track_id for track in lost_stracks]))
            print('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return [track for track in self.tracked_stracks if track.is_activated]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb