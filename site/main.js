(() => {
  const root = document.documentElement;
  const body = document.body;
  const store = {
    get(k, fallback) { try { return localStorage.getItem(k) ?? fallback; } catch { return fallback; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  const preferredLang = store.get("mqde-showcase-lang", (navigator.language || "").toLowerCase().startsWith("en") ? "en" : "zh");
  const preferredTheme = store.get("mqde-showcase-theme", "dark");
  const localPreview = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  const publicDocs = {
    zh: "https://github.com/zhipingwoods/MQDEsys/blob/main/docs/site/zh/index.md",
    en: "https://github.com/zhipingwoods/MQDEsys/blob/main/docs/site/en/index.md",
  };

  function syncLocalLinks(lang) {
    document.querySelectorAll("[data-local-only]").forEach((el) => {
      if (localPreview) {
        el.href = el.dataset.localHref || "#quickstart";
        el.removeAttribute("aria-disabled");
        el.removeAttribute("title");
        return;
      }
      el.href = "#quickstart";
      el.removeAttribute("target");
      el.setAttribute("aria-disabled", "true");
      el.title = lang === "zh" ? "本地启动服务后可用" : "Available after starting the local services";
      const text = lang === "zh" ? "启动后访问本地服务" : "Open after local startup";
      el.textContent = text;
    });
  }

  function setTheme(theme) {
    root.dataset.theme = theme;
    store.set("mqde-showcase-theme", theme);
    const btn = document.getElementById("theme-btn");
    if (btn) btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }

  function activeScreen() {
    return document.querySelector(".screen-tabs button.on");
  }

  function refreshCaption() {
    const lang = root.dataset.lang;
    const cap = document.getElementById("screen-caption");
    const active = activeScreen();
    if (cap && active) cap.textContent = active.dataset[`caption${lang === "zh" ? "Zh" : "En"}`] || "";
  }

  function setLang(lang) {
    root.dataset.lang = lang;
    root.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = lang === "zh" ? "MQDEsys · 模型量化部署评估系统" : "MQDEsys · Model Quantization & Deployment Evaluation";
    store.set("mqde-showcase-lang", lang);
    document.querySelectorAll("[data-zh]").forEach((el) => {
      const value = lang === "zh" ? el.dataset.zh : el.dataset.en;
      if (value != null) el.textContent = value;
    });
    document.querySelectorAll("[data-html-zh]").forEach((el) => {
      const value = lang === "zh" ? el.dataset.htmlZh : el.dataset.htmlEn;
      if (value != null) el.innerHTML = value;
    });
    document.querySelectorAll("[data-doc-link]").forEach((el) => {
      el.href = localPreview
        ? `../docs/site/_build/html/${lang}/index.html`
        : publicDocs[lang];
    });
    document.querySelectorAll("[data-doc-link-en]").forEach((el) => {
      el.href = localPreview
        ? "../docs/site/_build/html/en/index.html"
        : publicDocs.en;
    });
    document.querySelectorAll("[data-set-lang]").forEach((el) => {
      el.classList.toggle("on", el.dataset.setLang === lang);
    });
    refreshCaption();
    syncLocalLinks(lang);
  }

  document.querySelectorAll("[data-set-lang]").forEach((el) => {
    el.addEventListener("click", () => setLang(el.dataset.setLang));
  });
  document.getElementById("theme-btn")?.addEventListener("click", () => {
    setTheme(root.dataset.theme === "dark" ? "light" : "dark");
  });

  // Screenshot tabs.
  const tabs = [...document.querySelectorAll(".screen-tabs button")];
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((x) => x.classList.toggle("on", x === tab));
      const img = document.getElementById("screen-img");
      const shot = document.getElementById("screen-shot");
      if (img) img.src = tab.dataset.screen;
      if (shot) shot.dataset.full = tab.dataset.screen;
      refreshCaption();
    });
  });

  // Lightbox for every screenshot.
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  function closeLightbox() {
    lb?.classList.remove("open");
    lb?.setAttribute("aria-hidden", "true");
  }
  document.querySelectorAll(".shot").forEach((shot) => {
    shot.addEventListener("click", () => {
      const src = shot.dataset.full || shot.querySelector("img")?.src;
      if (!src || !lb || !lbImg) return;
      lbImg.src = src;
      lb.classList.add("open");
      lb.setAttribute("aria-hidden", "false");
    });
  });
  lb?.addEventListener("click", (e) => { if (e.target === lb) closeLightbox(); });
  document.querySelector(".lightbox-close")?.addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });

  // Reveal sections.
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          io.unobserve(entry.target);
        }
      }
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
  } else {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("visible"));
  }

  // Copy quick-start code.
  const copyBtn = document.getElementById("copy-btn");
  copyBtn?.addEventListener("click", async () => {
    const code = document.getElementById("code")?.textContent ?? "";
    const lang = root.dataset.lang;
    try {
      await navigator.clipboard.writeText(code);
      copyBtn.textContent = lang === "zh" ? "已复制" : "Copied";
    } catch {
      copyBtn.textContent = lang === "zh" ? "复制失败" : "Copy failed";
    }
    setTimeout(() => setLang(root.dataset.lang), 1400);
  });

  // Initial state.
  setTheme(preferredTheme);
  setLang(preferredLang);
})();
