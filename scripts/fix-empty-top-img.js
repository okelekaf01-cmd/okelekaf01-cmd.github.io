/* eslint-disable no-param-reassign */

// AnZhiYu theme treats `top_img: undefined` as "enabled" and renders a blank
// post header image (`src=""`), which makes the nav/title unreadable in light mode.
// If a post has no `top_img`, `cover`, or `randomcover`, force `top_img: false`
// so the theme uses the `not-top-img` header style.

hexo.extend.filter.register("before_post_render", function (data) {
  if (!data || data.layout !== "post") return data;

  // Respect explicit user intent.
  if (data.top_img === false) return data;

  const hasNonEmptyString = (v) => typeof v === "string" && v.trim().length > 0;

  const hasTopImg = hasNonEmptyString(data.top_img);
  const hasCover = hasNonEmptyString(data.cover) || hasNonEmptyString(data.randomcover);

  // When there's no usable image source, disable the top image for this post.
  if (!hasTopImg && !hasCover) {
    data.top_img = false;
  }

  return data;
});

