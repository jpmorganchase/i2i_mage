import os
import glob
import argparse
from i2i_cleanfid import fid
from metric import inception_score, BaseDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dst', type=str, help='Generated images directory')
    parser.add_argument('--task', type=str, default='inpaint', help='task for image generation')
    parser.add_argument('--data_root', type=str, default='../dataset/imagenet1k/val/', help='folder of dataset')
    args = parser.parse_args()


    if args.task == 'outpaint':
        modes_list = ['outpaint']
        ratio_list = list(range(1,14))
    elif args.task == 'inpaint':
        ratio_list = [4, 8]
        post_str = args.task+'_'
        modes_list =  ['center'] #, 'left', 'right', 'up', 'down']

    for mode  in modes_list:
        for ratio in ratio_list:
            post_str = mode+'_'+str(ratio/16.0)+'_'
            dst = glob.glob(os.path.join(args.dst, 'retain/*/{}ILSVRC2012_*.JPEG'.format(post_str)))
            src = [img.replace( os.path.join(args.dst, 'retain/'), args.data_root).replace(post_str, '') for img in dst]

            is_mean, is_std = inception_score(BaseDataset(dst), cuda=True, batch_size=2, resize=True, splits=10)
            fid_score = fid.compute_fid(src, dst, fdir2_is_image_bgr=True)
            metric=[os.path.join(args.dst, post_str), fid_score, is_mean, is_std]
            print('FID: {}'.format(fid_score))
            print('IS:{} {}'.format(is_mean, is_std))

            dst = glob.glob(os.path.join(args.dst, 'forget/*/{}ILSVRC2012_*.JPEG'.format(post_str)))
            src = [img.replace( os.path.join(args.dst, 'forget/'), args.data_root).replace(post_str, '') for img in dst]

            is_mean, is_std = inception_score(BaseDataset(dst), cuda=True, batch_size=2, resize=True, splits=10)
            fid_score = fid.compute_fid(src, dst, fdir2_is_image_bgr=True)
            
            print('FID: {}'.format(fid_score))
            print('IS:{} {}'.format(is_mean, is_std))
            metric +=[fid_score, is_mean, is_std]
            print('\t'.join(str(a) for a in metric), file=open('fid_is_eval_{}.csv'.format(args.task), 'a+'))


