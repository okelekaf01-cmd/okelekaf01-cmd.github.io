/*
 * Keep AnZhiYu aside widget order consistent (tracked in repo) even though the
 * theme is installed in `node_modules/` (which is gitignored).
 *
 * What we want on both posts and pages:
 * - Latest posts (最近发布)
 * - Categories (分类)
 * - Tags cloud (标签)
 * - Archives (归档)
 * - Site webinfo (运行时间/UV/PV 等)
 *
 * The stock theme only shows `card_recent_post` on posts, and hides categories/
 * tags/archives/webinfo while reading. This patch makes the sidebar richer and
 * aligns it with a “Butterfly-style” information density.
 */

const fs = require("fs");
const path = require("path");

function ensureFile(filePath, desired) {
  let current = "";
  try {
    current = fs.readFileSync(filePath, "utf8");
  } catch {
    return false;
  }

  if (current === desired) return true;

  fs.writeFileSync(filePath, desired, "utf8");
  return true;
}

hexo.on("ready", () => {
  const widgetIndex = path.join(hexo.theme_dir, "layout", "includes", "widget", "index.pug");

  const desired = `#aside-content.aside-content
  //- post
  if is_post()
    - const tocStyle = page.toc_style_simple
    - const tocStyleVal = tocStyle === true || tocStyle === false ? tocStyle : theme.toc.style_simple
    if showToc && tocStyleVal
      .sticky_layout
        include ./card_post_toc.pug
        !=partial('includes/widget/card_recent_post', {}, {cache: true})
        !=partial('includes/widget/card_categories', {}, {cache: true})
        .card-widget
          !=partial('includes/widget/card_ad', {}, {cache: true})
          !=partial('includes/widget/card_tags', {}, {cache: true})
          !=partial('includes/widget/card_archives', {}, {cache: true})
          !=partial('includes/widget/card_webinfo', {}, {cache: true})
        !=partial('includes/widget/card_bottom_self', {}, {cache: true})
    else
      !=partial('includes/widget/card_author', {}, {cache: true})
      !=partial('includes/widget/card_announcement', {}, {cache: true})
      !=partial('includes/widget/card_weixin', {}, {cache: true})
      !=partial('includes/widget/card_top_self', {}, {cache: true})
      .sticky_layout
        if showToc
          include ./card_post_toc.pug
        !=partial('includes/widget/card_recent_post', {}, {cache: true})
        !=partial('includes/widget/card_categories', {}, {cache: true})
        .card-widget
          !=partial('includes/widget/card_ad', {}, {cache: true})
          !=partial('includes/widget/card_tags', {}, {cache: true})
          !=partial('includes/widget/card_archives', {}, {cache: true})
          !=partial('includes/widget/card_webinfo', {}, {cache: true})
        !=partial('includes/widget/card_bottom_self', {}, {cache: true})
  else
    //- page
    !=partial('includes/widget/card_author', {}, {cache: true})
    !=partial('includes/widget/card_announcement', {}, {cache: true})
    !=partial('includes/widget/card_weixin', {}, {cache: true})
    !=partial('includes/widget/card_top_self', {}, {cache: true})
    !=partial('includes/widget/card_recent_post', {}, {cache: true})
    !=partial('includes/widget/card_categories', {}, {cache: true})  

    .sticky_layout
      if showToc
        include ./card_post_toc.pug
      .card-widget
        !=partial('includes/widget/card_ad', {}, {cache: true})
        !=partial('includes/widget/card_tags', {}, {cache: true})
        !=partial('includes/widget/card_archives', {}, {cache: true})
        !=partial('includes/widget/card_webinfo', {}, {cache: true})
      !=partial('includes/widget/card_bottom_self', {}, {cache: true})
`;

  const ok = ensureFile(widgetIndex, desired);
  if (!ok) {
    hexo.log.warn(`[patch-anzhiyu] sidebar widget index not found: ${widgetIndex}`);
    return;
  }

  hexo.log.info("[patch-anzhiyu] ensured aside widget order");
});

