const fs = require('fs');
const path = require('path');

const OUT_DIR = path.join(process.cwd(), 'source', 'img', 'covers');

const CARD = { x: 1040, y: 150, w: 320, h: 320, r: 56 };
const TEXT_X = 120;
const TITLE_Y_1 = 318;
const TITLE_Y_2 = 446;
const SUBTITLE_Y = 548;
const BADGE_Y = 642;
const BADGE_X = 120;

const commonStyle = {
  width: 1600,
  height: 900,
  rx: 38,
};

const covers = [
  {
    file: 'blog2-preset-skill-mcp.svg',
    colors: ['#3C7CFF', '#7D56FF'],
    glow: { x: 1180, y: 180, color: '#C7D2FF', opacity: 0.22, rx: 420, ry: 240 },
    title: ['Skill', 'MCP'],
    subtitle: 'Preset Code / Skill / MCP',
    badge: 'OpsMind Series',
    accent: '#EEF2FF',
    shadow: '#21306E',
    icon: 'stackBars',
  },
  {
    file: 'blog3-chart-recommender.svg',
    colors: ['#16C1B9', '#2B6CFF'],
    glow: { x: 420, y: 160, color: '#D2FFFF', opacity: 0.20, rx: 430, ry: 230 },
    title: ['Chart', 'Agent'],
    subtitle: 'Questioning / Recommendation / Reflection',
    badge: 'Visualization Notes',
    accent: '#EFFFFF',
    shadow: '#18436A',
    icon: 'lineChart',
  },
  {
    file: 'blog4-intent-recognition.svg',
    colors: ['#0FBF9F', '#1E90FF'],
    glow: { x: 1180, y: 170, color: '#D6FFF8', opacity: 0.18, rx: 420, ry: 210 },
    title: ['Intent', 'System'],
    subtitle: 'Recognition / Classification / Routing',
    badge: 'NLP Notes',
    accent: '#E8FFFE',
    shadow: '#0E4C67',
    icon: 'routing',
  },
  {
    file: 'blog5-llm-implementation.svg',
    colors: ['#6046FF', '#FF6A88'],
    glow: { x: 1190, y: 175, color: '#FFD1EA', opacity: 0.24, rx: 430, ry: 230 },
    title: ['LLM', 'Stack'],
    subtitle: 'Prompt / Service / Optimization / Analysis',
    badge: 'OpsMind LLM',
    accent: '#FFF1F7',
    shadow: '#3B247A',
    icon: 'bot',
  },
  {
    file: 'blog6-chart-refactor.svg',
    colors: ['#30C39E', '#2E86FF'],
    glow: { x: 430, y: 170, color: '#D8FFF0', opacity: 0.18, rx: 440, ry: 220 },
    title: ['Chart', 'Refactor'],
    subtitle: 'Config / Decomposition / Maintainability',
    badge: 'Refactor Log',
    accent: '#F1FFFD',
    shadow: '#1B4E68',
    icon: 'refactorBars',
  },
  {
    file: 'blog7-engineering-milestone.svg',
    colors: ['#3B82F6', '#0EA5A5'],
    glow: { x: 1160, y: 180, color: '#D7F6FF', opacity: 0.18, rx: 410, ry: 210 },
    title: ['Engineering', 'Milestone'],
    subtitle: 'Testing / UI / Tooling / Workflow',
    badge: 'Project Progress',
    accent: '#F2FFFF',
    shadow: '#224E68',
    icon: 'milestoneBars',
    subtitleFontSize: 46,
    subtitleLength: 770,
  },
  {
    file: 'blog8-preprocessor-upgrade.svg',
    colors: ['#24C6A2', '#2F80ED'],
    glow: { x: 1190, y: 200, color: '#D8FFF8', opacity: 0.18, rx: 390, ry: 220 },
    title: ['Data', 'Prep'],
    subtitle: 'Cleaning / Structuring / Inference / Upgrade',
    badge: 'Data Upgrade',
    accent: '#F1FFFD',
    shadow: '#1C4C67',
    icon: 'dataPrep',
  },
  {
    file: 'blog9-technical-discoveries.svg',
    colors: ['#5B4CFF', '#FF6A88'],
    glow: { x: 420, y: 700, color: '#FFD3E5', opacity: 0.18, rx: 470, ry: 250 },
    title: ['Tech', 'Discovery'],
    subtitle: 'AI Architecture / Agent / Tool Calling',
    badge: 'Architecture Notes',
    accent: '#FFF1F7',
    shadow: '#39256F',
    icon: 'network',
  },
  {
    file: 'first-post-blog-launch.svg',
    colors: ['#4F46E5', '#22C55E'],
    glow: { x: 1180, y: 170, color: '#DFFFEA', opacity: 0.18, rx: 430, ry: 220 },
    title: ['Blog', 'Launch'],
    subtitle: 'Hugo / GitHub Pages / From 404 to Online',
    badge: 'First Story',
    accent: '#F0FFF7',
    shadow: '#253B5C',
    icon: 'document',
  },
  {
    file: 'internship-motion-library-day1-day2.svg',
    colors: ['#4ED3A3', '#5B9BFF'],
    glow: { x: 430, y: 170, color: '#B4FFF1', opacity: 0.22, rx: 470, ry: 240 },
    title: ['Motion', 'Library'],
    subtitle: 'Architecture / Parameters / Layout / Debugging',
    badge: 'Internship Notes',
    accent: '#ECFFFB',
    shadow: '#173B54',
    icon: 'motionBot',
  },
  {
    file: 'studio-motion-devlog-2026-03-19.svg',
    colors: ['#5843FF', '#FF5D8E'],
    glow: { x: 1180, y: 170, color: '#FFD1EA', opacity: 0.24, rx: 430, ry: 230 },
    title: ['Studio', 'Motion'],
    subtitle: 'Crash Fix / UI Cleanup / Positioning / Interaction',
    badge: 'Dev Log',
    accent: '#FFF0F7',
    shadow: '#331C78',
    icon: 'studioBot',
  },
];

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function wrap(lines) {
  return lines.map((line, i) => {
    const y = i === 0 ? TITLE_Y_1 : TITLE_Y_2;
    return `<text x="${TEXT_X}" y="${y}" fill="#FFFFFF" font-size="128" font-family="Arial, Helvetica, sans-serif" font-weight="800" letter-spacing="-5">${esc(line)}</text>`;
  }).join('\n  ');
}

function badge(text) {
  const width = Math.max(220, Math.min(340, 140 + text.length * 12));
  return `
  <rect x="${BADGE_X}" y="${BADGE_Y}" width="${width}" height="64" rx="32" fill="rgba(255,255,255,0.18)"/>
  <text x="${BADGE_X + 44}" y="${BADGE_Y + 43}" fill="#FFFFFF" font-size="34" font-family="Arial, Helvetica, sans-serif" font-weight="700">${esc(text)}</text>`;
}

function subtitleText(cover) {
  const fontSize = cover.subtitleFontSize || 48;
  const textLength = cover.subtitleLength || 790;
  return `<text x="126" y="${SUBTITLE_Y}" fill="${cover.accent}" font-size="${fontSize}" font-family="Arial, Helvetica, sans-serif" font-weight="600" textLength="${textLength}" lengthAdjust="spacingAndGlyphs">${esc(cover.subtitle)}</text>`;
}

function iconFrame(id) {
  return `
  <g filter="url(#shadow-${id})">
    <rect x="${CARD.x}" y="${CARD.y}" width="${CARD.w}" height="${CARD.h}" rx="${CARD.r}" fill="#FFFDFC"/>
  </g>`;
}

function iconInnerBg() {
  return `<rect x="${CARD.x + 34}" y="${CARD.y + 34}" width="${CARD.w - 68}" height="${CARD.h - 68}" rx="34" fill="#EAF4FF" opacity="0.66"/>`;
}

function drawIcon(icon, start, end) {
  const left = CARD.x;
  const top = CARD.y;
  switch (icon) {
    case 'stackBars':
      return `
  ${iconFrame(icon)}
  <rect x="${left + 56}" y="${top + 72}" width="208" height="56" rx="18" fill="${start}"/>
  <rect x="${left + 56}" y="${top + 144}" width="208" height="56" rx="18" fill="${end}"/>
  <rect x="${left + 56}" y="${top + 216}" width="208" height="56" rx="18" fill="#182E57"/>`;
    case 'lineChart':
      return `
  ${iconFrame(icon)}
  <path d="M${left + 58} ${top + 244}L${left + 116} ${top + 158}L${left + 174} ${top + 212}L${left + 240} ${top + 108}" stroke="${end}" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="${left + 58}" cy="${top + 244}" r="16" fill="${start}"/>
  <circle cx="${left + 116}" cy="${top + 158}" r="16" fill="${start}"/>
  <circle cx="${left + 174}" cy="${top + 212}" r="16" fill="${start}"/>
  <circle cx="${left + 240}" cy="${top + 108}" r="16" fill="${start}"/>`;
    case 'routing':
      return `
  ${iconFrame(icon)}
  <circle cx="${left + 160}" cy="${top + 104}" r="56" fill="${start}" opacity="0.24"/>
  <rect x="${left + 72}" y="${top + 176}" width="176" height="22" rx="11" fill="${end}"/>
  <rect x="${left + 72}" y="${top + 224}" width="126" height="22" rx="11" fill="${start}"/>
  <rect x="${left + 72}" y="${top + 272}" width="212" height="22" rx="11" fill="#1D3557"/>`;
    case 'bot':
      return `
  ${iconFrame(icon)}
  <circle cx="${left + 160}" cy="${top + 160}" r="92" fill="#4E32C0"/>
  <path d="M${left + 116} ${top + 120}H${left + 192}L${left + 174} ${top + 156}H${left + 208}V${top + 230}H${left + 108}V${top + 156}H${left + 136}L${left + 116} ${top + 120}Z" fill="#FFC341"/>
  <rect x="${left + 176}" y="${top + 122}" width="28" height="108" rx="8" fill="#FF4D7A"/>
  <rect x="${left + 135}" y="${top + 164}" width="28" height="18" rx="9" fill="#FFFFFF"/>
  <rect x="${left + 175}" y="${top + 164}" width="28" height="18" rx="9" fill="#FFFFFF"/>`;
    case 'refactorBars':
      return `
  ${iconFrame(icon)}
  <path d="M${left + 74} ${top + 236}H${left + 244}" stroke="#1F4EFF" stroke-width="18" stroke-linecap="round"/>
  <path d="M${left + 74} ${top + 180}H${left + 206}" stroke="${start}" stroke-width="18" stroke-linecap="round"/>
  <path d="M${left + 74} ${top + 124}H${left + 164}" stroke="#19334F" stroke-width="18" stroke-linecap="round"/>
  <rect x="${left + 60}" y="${top + 104}" width="22" height="152" rx="11" fill="#EAF6FF"/>`;
    case 'milestoneBars':
      return `
  ${iconFrame(icon)}
  <rect x="${left + 66}" y="${top + 72}" width="60" height="190" rx="18" fill="${end}"/>
  <rect x="${left + 142}" y="${top + 138}" width="60" height="124" rx="18" fill="${start}"/>
  <rect x="${left + 218}" y="${top + 100}" width="60" height="162" rx="18" fill="#173B58"/>`;
    case 'dataPrep':
      return `
  ${iconFrame(icon)}
  <rect x="${left + 64}" y="${top + 76}" width="192" height="38" rx="12" fill="#173A57"/>
  <rect x="${left + 64}" y="${top + 132}" width="192" height="38" rx="12" fill="${start}"/>
  <rect x="${left + 64}" y="${top + 188}" width="192" height="38" rx="12" fill="${end}"/>
  <circle cx="${left + 268}" cy="${top + 268}" r="34" fill="rgba(255,255,255,0.16)"/>`;
    case 'network':
      return `
  ${iconFrame(icon)}
  <circle cx="${left + 162}" cy="${top + 78}" r="42" fill="#5B4CFF" opacity="0.24"/>
  <circle cx="${left + 102}" cy="${top + 194}" r="42" fill="#FF6A88" opacity="0.24"/>
  <circle cx="${left + 220}" cy="${top + 194}" r="42" fill="#FFC341" opacity="0.32"/>
  <path d="M${left + 162} ${top + 120}L${left + 102} ${top + 150}" stroke="#5B4CFF" stroke-width="12" stroke-linecap="round"/>
  <path d="M${left + 162} ${top + 120}L${left + 220} ${top + 150}" stroke="#FF6A88" stroke-width="12" stroke-linecap="round"/>`;
    case 'document':
      return `
  ${iconFrame(icon)}
  <rect x="${left + 80}" y="${top + 74}" width="160" height="184" rx="28" fill="#EEF5FF"/>
  <path d="M${left + 116} ${top + 118}H${left + 208}" stroke="#4F46E5" stroke-width="16" stroke-linecap="round"/>
  <path d="M${left + 116} ${top + 170}H${left + 208}" stroke="#22C55E" stroke-width="16" stroke-linecap="round"/>
  <path d="M${left + 116} ${top + 222}H${left + 180}" stroke="#18334C" stroke-width="16" stroke-linecap="round"/>`;
    case 'motionBot':
      return `
  ${iconFrame(icon)}
  <rect x="${left + 34}" y="${top + 34}" width="252" height="252" rx="36" fill="url(#bg-inner-motionBot)" opacity="0.16"/>
  <rect x="${left + 90}" y="${top + 82}" width="84" height="84" rx="20" fill="#1FAF8D"/>
  <rect x="${left + 188}" y="${top + 82}" width="84" height="84" rx="20" fill="#4B6DFF"/>
  <rect x="${left + 90}" y="${top + 192}" width="182" height="78" rx="18" fill="#163D58" opacity="0.86"/>
  <rect x="${left + 126}" y="${top + 118}" width="12" height="12" rx="6" fill="#FFFFFF"/>
  <rect x="${left + 224}" y="${top + 118}" width="12" height="12" rx="6" fill="#FFFFFF"/>
  <rect x="${left + 126}" y="${top + 227}" width="112" height="12" rx="6" fill="#FFFFFF" opacity="0.92"/>`;
    case 'studioBot':
      return `
  ${iconFrame(icon)}
  <circle cx="${left + 160}" cy="${top + 160}" r="98" fill="#4E32C0"/>
  <path d="M${left + 116} ${top + 110}H${left + 202}L${left + 181} ${top + 149}H${left + 218}V${top + 218}H${left + 98}V${top + 149}H${left + 138}L${left + 116} ${top + 110}Z" fill="#FFC341"/>
  <rect x="${left + 188}" y="${top + 112}" width="30" height="106" rx="8" fill="#FF4D7A"/>
  <rect x="${left + 136}" y="${top + 153}" width="28" height="18" rx="9" fill="#FFFFFF"/>
  <rect x="${left + 176}" y="${top + 153}" width="28" height="18" rx="9" fill="#FFFFFF"/>`;
    default:
      return `\n  ${iconFrame(icon)}`;
  }
}

function coverSvg(cover) {
  const [start, end] = cover.colors;
  return `<svg width="${commonStyle.width}" height="${commonStyle.height}" viewBox="0 0 ${commonStyle.width} ${commonStyle.height}" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="40" y1="20" x2="1560" y2="880" gradientUnits="userSpaceOnUse">
      <stop stop-color="${start}"/>
      <stop offset="1" stop-color="${end}"/>
    </linearGradient>
    <linearGradient id="bg-inner-motionBot" x1="${CARD.x + 34}" y1="${CARD.y + 34}" x2="${CARD.x + CARD.w - 34}" y2="${CARD.y + CARD.h - 34}" gradientUnits="userSpaceOnUse">
      <stop stop-color="${start}"/>
      <stop offset="1" stop-color="${end}"/>
    </linearGradient>
    <radialGradient id="glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(${cover.glow.x} ${cover.glow.y}) rotate(15) scale(${cover.glow.rx} ${cover.glow.ry})">
      <stop stop-color="${cover.glow.color}" stop-opacity="${cover.glow.opacity}"/>
      <stop offset="1" stop-color="${cover.glow.color}" stop-opacity="0"/>
    </radialGradient>
    <filter id="shadow-${cover.icon}" x="0" y="0" width="1600" height="900" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
      <feDropShadow dx="0" dy="20" stdDeviation="18" flood-color="${cover.shadow}" flood-opacity="0.22"/>
    </filter>
  </defs>
  <rect width="${commonStyle.width}" height="${commonStyle.height}" rx="${commonStyle.rx}" fill="url(#bg)"/>
  <rect width="${commonStyle.width}" height="${commonStyle.height}" rx="${commonStyle.rx}" fill="url(#glow)"/>
  <g opacity="0.10" stroke="#FFFFFF">
    <path d="M0 200H1600"/>
    <path d="M0 520H1600"/>
    <path d="M320 0V900"/>
    <path d="M1120 0V900"/>
  </g>
  ${wrap(cover.title)}
  ${subtitleText(cover)}
  ${badge(cover.badge)}
  ${drawIcon(cover.icon, start, end)}
</svg>`;
}

for (const cover of covers) {
  fs.writeFileSync(path.join(OUT_DIR, cover.file), coverSvg(cover));
}
