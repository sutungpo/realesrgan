## normal data process:
  ### dataloader
  ### prefetcher next:
    gt image
    horizontal flips or rotate augment
    crop or pad to 400
  ### model.feed_data:
    two-order degradations:
      blur, resize, noise, JPEG compression, blur, resize, noise, [[resize back lq + sinc filter] + JPEG compression or JPEG compression + [resize back lq + sinc filter]]
    random crop gt to gt_size(256) and lq to gt_size / scale

## paired data process:
  ### dataloader
  ### prefetcher next:
    gt image, lq image
    random crop gt to gt_size(256) and lq to gt_size / scale
    horizontal flips or rotate augment
  ### model.feed_data:
    gt image, lq image

