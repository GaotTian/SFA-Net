import time
import cv2
from segment_anything import sam_model_registry
from automatic_mask_and_probability_generator import SamAutomaticMaskAndProbabilityGenerator
from model import matches, Matcher
def showAndsaveMatches(points0, points1, img0, img1,windowname, flag,textflag,text, savepath):
    """
    :param points0: the keypoints of the image0
    :param points1: the keypoints of the image1
    :param img0: the image0
    :param img1: the image1
    :param path: the path of the saving image
    :return:None
    """
    Nums = points0.shape[0]
    # print(Nums)

    kp1, kp2, matches = [], [], []
    if flag == 0:
        for i in range(Nums):
            kp1.append(cv2.KeyPoint(float(points0[i][0]), float(points0[i][1]), 1))
            kp2.append(cv2.KeyPoint(float(points1[i][0]), float(points1[i][1]), 1))
            matches.append(cv2.DMatch(i, i, 1))
    else:#Fast
        for i in range(Nums):
            kp1.append(cv2.KeyPoint(float(points0[i][1]), float(points0[i][0]), 1))
            kp2.append(cv2.KeyPoint(float(points1[i][1]), float(points1[i][0]), 1))
            matches.append(cv2.DMatch(i, i, 1))
    img0 = cv2.drawKeypoints(img0,kp1,None,color=(0,0,255))
    img1 = cv2.drawKeypoints(img1,kp2,None,color=(0,255,0))
    Match = cv2.drawMatches(img0, kp1, img1, kp2, matches, None,matchColor=(0,255,255), flags=4)
    if textflag == True:
        sc = min(img0.shape[0] / 800., 2.0)  # Big text.
        Ht = int(40 * sc)  # text height
        txt_color_fg = (255, 255, 255)
        txt_color_bg = (0, 0, 0)
        text = text
        for i, t in enumerate(text):
            cv2.putText(Match, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
            cv2.putText(Match, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    cv2.imshow(windowname, Match)

    if savepath is not None:
        current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        cv2.imwrite(savepath + windowname + current_time + '.png', Match)
        print(savepath + str(current_time) + '.png')
    cv2.waitKey(0)


if __name__ == "__main__":
    i = 3 # 44 46
    img_filename0 = 'D:/code_gt/SFA-Net/usedata/infrared-infrared/{}-1.jpg'.format(i)
    img_filename1 = 'D:/code_gt/SFA-Net/usedata/infrared-infrared/{}-2.jpg'.format(i)

    img0 = cv2.imread(img_filename0)
    img1 = cv2.imread(img_filename1)

    print('*  Load model...')
    config = {'detector': {'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': 3000, 'remove_borders': 4, 'cuda': 1},
              'matcher': {'sinkhorn_iterations': 120, 'match_threshold': 0.4, 'cuda': 1}}
    kernel = 3
    risg = Matcher(config)

    sam = sam_model_registry["default"](checkpoint=".\weights/sam_vit.pth").to(device="cuda")
    sam = SamAutomaticMaskAndProbabilityGenerator(sam)
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('.\weights\model.yml.gz')

    print('*  Load model complete...')

    t0 = time.perf_counter()
    nim, mmkpts0, mmkpts1, Affine = matches(img0,img1,risg,sam,kernel,edge_detection)
    t1 = time.perf_counter()


    print('*  Matching complete...')

    print('time:',round(t1-t0,2),'s')
    showAndsaveMatches(mmkpts0, mmkpts1, img0, img1, 'sam', 0, 1, [], None)