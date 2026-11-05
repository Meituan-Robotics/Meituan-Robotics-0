(() => {
  "use strict";

  const sectionIds = [
    "overview",
    "data",
    "method",
    "evaluation",
    "demos",
    "conclusion",
    "citation",
  ];
  const projectNav = document.querySelector('nav[aria-label="Project sections"]');
  const progress = document.querySelector('.reading-progress[role="progressbar"]');
  const progressBar = progress?.querySelector("span");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let navigationFrame = 0;
  let currentActiveId = null;

  function updateNavigation() {
    navigationFrame = 0;
    const anchor = Math.min(window.innerHeight * 0.35, 180);
    let activeId = sectionIds[0];

    for (const id of sectionIds) {
      const section = document.getElementById(id);
      if (section && section.getBoundingClientRect().top <= anchor) activeId = id;
    }

    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const readingProgress = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
    if (readingProgress >= 99.9) activeId = sectionIds.at(-1);

    projectNav?.querySelectorAll("a").forEach((link) => {
      if (link.getAttribute("href") === `#${activeId}`) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    });

    if (projectNav && activeId !== currentActiveId) {
      const activeLink = projectNav.querySelector('a[aria-current="location"]');
      if (activeLink) {
        const left = activeLink.offsetLeft - (projectNav.clientWidth - activeLink.offsetWidth) / 2;
        projectNav.scrollTo({
          left: Math.max(0, left),
          behavior: reducedMotion ? "auto" : "smooth",
        });
      }
      currentActiveId = activeId;
    }

    const clampedProgress = Math.min(100, Math.max(0, readingProgress));
    if (progressBar) progressBar.style.width = `${clampedProgress}%`;
    progress?.setAttribute("aria-valuenow", String(Math.round(clampedProgress)));
  }

  function scheduleNavigationUpdate() {
    if (navigationFrame) return;
    navigationFrame = window.requestAnimationFrame(updateNavigation);
  }

  updateNavigation();
  window.addEventListener("scroll", scheduleNavigationUpdate, { passive: true });
  window.addEventListener("resize", scheduleNavigationUpdate);

  document.querySelectorAll(".video-player select").forEach((select) => {
    select.addEventListener("change", () => {
      const video = select.closest(".video-player")?.querySelector("video");
      const playbackRate = Number(select.value);
      if (!video || !Number.isFinite(playbackRate)) return;
      video.playbackRate = playbackRate;
      video.defaultPlaybackRate = playbackRate;
    });
  });

  const citationButton = document.querySelector(".copy-citation");
  const citationText = document.querySelector("#citation code")?.textContent?.trim();
  let citationResetTimer = 0;

  async function writeClipboard(text) {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) throw new Error("Clipboard API unavailable");
  }

  citationButton?.addEventListener("click", async () => {
    if (!citationText) return;
    window.clearTimeout(citationResetTimer);

    try {
      await writeClipboard(citationText);
      citationButton.dataset.status = "copied";
      citationButton.textContent = "Copied";
    } catch {
      citationButton.dataset.status = "error";
      citationButton.textContent = "Copy failed";
    }

    citationResetTimer = window.setTimeout(() => {
      citationButton.dataset.status = "idle";
      citationButton.textContent = "Copy Citation";
    }, 2000);
  });

  const dataFigure = document.querySelector(".robot-data-figure");
  const sourceControls = dataFigure?.querySelectorAll(".source-control, .donut-hit") ?? [];
  let lockedSource = null;

  function setActiveSource(source) {
    if (!dataFigure) return;
    if (source) dataFigure.dataset.activeSource = source;
    else delete dataFigure.dataset.activeSource;
  }

  function setPressedSource(source) {
    sourceControls.forEach((control) => {
      control.setAttribute("aria-pressed", String(control.dataset.source === source));
    });
  }

  sourceControls.forEach((control) => {
    const source = control.dataset.source;
    if (!source) return;

    control.addEventListener("pointerenter", () => setActiveSource(source));
    control.addEventListener("pointerleave", () => setActiveSource(lockedSource));
    control.addEventListener("focus", () => setActiveSource(source));
    control.addEventListener("blur", () => setActiveSource(lockedSource));
    control.addEventListener("click", () => {
      lockedSource = lockedSource === source ? null : source;
      setPressedSource(lockedSource);
      setActiveSource(lockedSource);
    });

    if (control.matches("svg [role='button']")) {
      control.addEventListener("keydown", (event) => {
        if (event.repeat || (event.key !== "Enter" && event.key !== " ")) return;
        event.preventDefault();
        control.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      });
    }
  });

  const resultGrid = document.querySelector(".result-card-grid");
  if (resultGrid && !reducedMotion && "IntersectionObserver" in window) {
    resultGrid.classList.add("is-animated");
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        resultGrid.classList.add("is-revealed");
        observer.disconnect();
      },
      { threshold: 0.2 },
    );
    observer.observe(resultGrid);
  } else {
    resultGrid?.classList.add("is-revealed");
  }
})();
