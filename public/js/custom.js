(function () {
  const HOME_WELCOME_SESSION_KEY = "home-welcome-dismissed-v4";
  const HOME_WELCOME_MESSAGES = [
    "欢迎光临，先别急着走，我这次真的写了点东西。",
    "来都来了，不如看看我最近又折腾出了什么。",
    "这里偶尔有技术，偶尔有灵感，偶尔还有一点倔强。",
    "如果你也是被好奇心骗进来的，那我们算同伙。",
    "本博客不保证顿悟，但尽量保证不无聊。",
    "今天的随机任务：从这里带走一个新想法。",
    "欢迎来到现场，这里正在稳定产出一些认真折腾。",
    "别紧张，这不是考试，只是一起随便看看。",
  ];

  const INTRO_DISMISS_MS = 560;
  const state = {
    introEl: null,
    welcomeTextEl: null,
    typingTimer: null,
    typingDelayTimer: null,
    dismissTimer: null,
    isDismissing: false,
    touchStartY: null,
  };

  const removeThemePopup = () => {
    document.getElementById("popup-window")?.remove();
  };

  const clearTypingTimers = () => {
    if (state.typingTimer) {
      window.clearInterval(state.typingTimer);
      state.typingTimer = null;
    }

    if (state.typingDelayTimer) {
      window.clearTimeout(state.typingDelayTimer);
      state.typingDelayTimer = null;
    }

    if (state.dismissTimer) {
      window.clearTimeout(state.dismissTimer);
      state.dismissTimer = null;
    }
  };

  const isHomePage = () =>
    Boolean(window.GLOBAL_CONFIG_SITE?.isHome || document.getElementById("home_top"));

  const hasDismissedIntro = () => {
    try {
      return window.sessionStorage.getItem(HOME_WELCOME_SESSION_KEY) === "1";
    } catch (error) {
      return false;
    }
  };

  const markIntroDismissed = () => {
    try {
      window.sessionStorage.setItem(HOME_WELCOME_SESSION_KEY, "1");
    } catch (error) {
      // Ignore storage failures and continue without persistence.
    }
  };

  const sampleWelcomeMessage = () => {
    const pool = [...HOME_WELCOME_MESSAGES];

    for (let i = pool.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }

    return pool[0] || "";
  };

  const setIntroState = (stateName) => {
    document.body?.classList.remove("site-home-intro-active", "site-home-intro-leaving");
    document.documentElement.classList.remove(
      "site-home-intro-active",
      "site-home-intro-leaving"
    );

    if (!stateName) return;

    document.body?.classList.add(stateName);
    document.documentElement.classList.add(stateName);
  };

  const startWelcomeTyping = (text) => {
    clearTypingTimers();

    if (!state.welcomeTextEl) return;
    if (!text) {
      state.welcomeTextEl.textContent = "";
      return;
    }

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    if (prefersReducedMotion) {
      state.welcomeTextEl.textContent = text;
      return;
    }

    state.welcomeTextEl.textContent = "";
    let index = 0;

    state.typingDelayTimer = window.setTimeout(() => {
      state.typingTimer = window.setInterval(() => {
        index += 1;
        state.welcomeTextEl.textContent = text.slice(0, index);

        if (index >= text.length) {
          window.clearInterval(state.typingTimer);
          state.typingTimer = null;
        }
      }, 68);
    }, 220);
  };

  const teardownIntro = () => {
    clearTypingTimers();
    state.introEl?.remove();
    state.introEl = null;
    state.welcomeTextEl = null;
    state.touchStartY = null;
    state.isDismissing = false;
    setIntroState(null);
  };

  const dismissIntro = () => {
    if (!state.introEl || state.isDismissing) return;

    state.isDismissing = true;
    clearTypingTimers();
    markIntroDismissed();
    setIntroState("site-home-intro-leaving");
    state.introEl.classList.add("is-leaving");

    state.dismissTimer = window.setTimeout(() => {
      teardownIntro();
    }, INTRO_DISMISS_MS);
  };

  const shouldTriggerDismiss = (deltaY) => deltaY > 10;

  const handleIntroWheel = (event) => {
    if (!state.introEl || state.isDismissing) return;

    if (shouldTriggerDismiss(event.deltaY)) {
      event.preventDefault();
      dismissIntro();
    }
  };

  const handleIntroTouchStart = (event) => {
    if (!state.introEl || state.isDismissing) return;
    state.touchStartY = event.touches[0]?.clientY ?? null;
  };

  const handleIntroTouchMove = (event) => {
    if (!state.introEl || state.isDismissing) return;

    const currentY = event.touches[0]?.clientY;
    if (currentY == null || state.touchStartY == null) return;

    if (state.touchStartY - currentY > 22) {
      event.preventDefault();
      dismissIntro();
    }
  };

  const handleIntroKeydown = (event) => {
    if (!state.introEl || state.isDismissing) return;

    if (["ArrowDown", "PageDown", " ", "Enter"].includes(event.key)) {
      event.preventDefault();
      dismissIntro();
    }
  };

  const initHomeWelcomeIntro = () => {
    teardownIntro();

    if (!isHomePage() || hasDismissedIntro()) return;

    const blogContainer = document.getElementById("blog-container");
    const homeTop = document.getElementById("home_top");
    if (!blogContainer || !homeTop) return;

    const introEl = document.createElement("section");
    introEl.id = "site-home-welcome-intro";
    introEl.className = "site-home-welcome-intro";
    introEl.setAttribute("aria-label", "首页开场动画");
    introEl.innerHTML = `
      <div class="site-home-welcome-intro__backdrop"></div>
      <div class="site-home-welcome-intro__inner">
        <div class="site-home-welcome-intro__avatar-wrap">
          <img
            class="site-home-welcome-intro__avatar"
            src="/img/avatar/profile-avatar-square.jpg"
            alt="wwxdsg 的头像"
          />
        </div>
        <p class="site-home-welcome-intro__welcome">
          <span class="site-home-welcome-intro__welcome-text"></span>
          <span class="site-home-welcome-intro__cursor" aria-hidden="true"></span>
        </p>
      </div>
    `;

    blogContainer.insertBefore(introEl, homeTop);

    state.introEl = introEl;
    state.welcomeTextEl = introEl.querySelector(
      ".site-home-welcome-intro__welcome-text"
    );

    setIntroState("site-home-intro-active");
    startWelcomeTyping(sampleWelcomeMessage());

    introEl.addEventListener("wheel", handleIntroWheel, { passive: false });
    introEl.addEventListener("touchstart", handleIntroTouchStart, { passive: true });
    introEl.addEventListener("touchmove", handleIntroTouchMove, { passive: false });

    window.requestAnimationFrame(() => {
      introEl.classList.add("is-visible");
    });
  };

  const initDailyQuote = () => {
    const quoteEl = document.getElementById("daily-quote");
    if (!quoteEl || quoteEl.dataset.loaded === "true") return;

    quoteEl.dataset.loaded = "true";
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
  };

  const initSiteEnhancements = () => {
    removeThemePopup();
    initDailyQuote();
    initHomeWelcomeIntro();
  };

  initSiteEnhancements();
  document.addEventListener("DOMContentLoaded", initSiteEnhancements);
  document.addEventListener("pjax:complete", initSiteEnhancements);
  document.addEventListener("keydown", handleIntroKeydown);
})();
