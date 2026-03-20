(function () {
  const removeThemePopup = () => {
    document.getElementById("popup-window")?.remove();
  };

  removeThemePopup();
  document.addEventListener("DOMContentLoaded", removeThemePopup);
  document.addEventListener("pjax:complete", removeThemePopup);

  const quoteEl = document.getElementById("daily-quote");
  if (!quoteEl) return;

  const fromEl = document.getElementById("daily-quote-from");

  // Hitokoto: https://hitokoto.cn/
  fetch("https://v1.hitokoto.cn/?encode=json", { cache: "no-store" })
    .then((r) => r.json())
    .then((data) => {
      if (!data || !data.hitokoto) throw new Error("invalid hitokoto payload");
      quoteEl.textContent = data.hitokoto;
      if (fromEl) {
        const from = data.from ? `出自 ${data.from}` : "";
        fromEl.textContent = from;
      }
    })
    .catch(() => {
      quoteEl.textContent = "今天也要保持好奇心。";
      if (fromEl) fromEl.textContent = "";
    });
})();
