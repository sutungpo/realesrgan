## normal data process:
  ### dataloader
  ### prefetcher next:
    gt image
    read as numpy BGR order, range to [0.0,1.0]
    horizontal flips or rotate augment
    crop or pad to 400
    BGR to RGB, HWC to CHW, numpy to tensor
  ### model.feed_data:
    two-order degradations:
      blur, resize, noise, JPEG compression, blur, resize, noise, [[resize back lq + sinc filter] + JPEG compression or JPEG compression + [resize back lq + sinc filter]]
    random crop gt to gt_size(256) and lq to gt_size / scale

## paired data process:
  ### dataloader
  ### prefetcher next:
    gt image, lq image
    read as numpy BGR order, range to [0.0,1.0]
    random crop gt to gt_size(256) and lq to gt_size / scale
    horizontal flips or rotate augment
    BGR to RGB, HWC to CHW, numpy to tensor
  ### model.feed_data:
    gt image, lq image

