// Shared by the React preview and the standalone GitHub Pages export.
(() => {
  "use strict";

  if (typeof window === "undefined" || typeof document === "undefined") return;
  if (window.__mr0FigureViewerInstalled) return;
  window.__mr0FigureViewerInstalled = true;

  let dialog;
  let viewport;
  let canvas;
  let title;
  let status;
  let fitButton;
  let zoomOutButton;
  let zoomInButton;
  let zoomLevel;
  let closeButton;
  let image = null;
  let trigger = null;
  let zoom = 1;
  let scrollLock = null;
  let frame = 0;
  const minimumZoom = 0.5;
  const maximumZoom = 4;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function button(text, label, onClick) {
    const node = element("button", "", text);
    node.type = "button";
    node.setAttribute("aria-label", label);
    node.addEventListener("click", onClick);
    return node;
  }

  function lockScroll() {
    scrollLock = {
      x: window.scrollX,
      y: window.scrollY,
      overflow: document.body.style.overflow,
      position: document.body.style.position,
      top: document.body.style.top,
      width: document.body.style.width,
      rootOverflow: document.documentElement.style.overflow,
    };
    document.body.style.overflow = "hidden";
    document.body.style.position = "fixed";
    document.body.style.top = `-${scrollLock.y}px`;
    document.body.style.width = "100%";
    document.documentElement.style.overflow = "hidden";
  }

  function unlockScroll() {
    if (!scrollLock) return;
    const saved = scrollLock;
    scrollLock = null;
    document.body.style.overflow = saved.overflow;
    document.body.style.position = saved.position;
    document.body.style.top = saved.top;
    document.body.style.width = saved.width;
    document.documentElement.style.overflow = saved.rootOverflow;
    const behavior = document.documentElement.style.scrollBehavior;
    document.documentElement.style.scrollBehavior = "auto";
    window.scrollTo(saved.x, saved.y);
    document.documentElement.style.scrollBehavior = behavior;
  }

  function updateImage(center = false) {
    if (!dialog?.open || !image?.naturalWidth || !image.naturalHeight) return;
    const availableWidth = Math.max(1, viewport.clientWidth - 48);
    const availableHeight = Math.max(1, viewport.clientHeight - 48);
    const fit = Math.min(availableWidth / image.naturalWidth, availableHeight / image.naturalHeight);
    image.style.width = `${image.naturalWidth * fit * zoom}px`;
    image.style.height = `${image.naturalHeight * fit * zoom}px`;
    zoomLevel.textContent = `${Math.round(zoom * 100)}%`;
    fitButton.setAttribute("aria-pressed", String(zoom === 1));
    zoomOutButton.disabled = zoom <= minimumZoom;
    zoomInButton.disabled = zoom >= maximumZoom;
    if (center) {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        if (!dialog.open) return;
        viewport.scrollLeft = Math.max(0, (viewport.scrollWidth - viewport.clientWidth) / 2);
        viewport.scrollTop = Math.max(0, (viewport.scrollHeight - viewport.clientHeight) / 2);
      });
    }
  }

  function setZoom(next) {
    zoom = Math.max(minimumZoom, Math.min(maximumZoom, next));
    updateImage(true);
  }

  function close() {
    if (dialog?.open) dialog.close();
  }

  function createDialog() {
    dialog = element("dialog", "figure-lightbox");
    dialog.setAttribute("aria-labelledby", "figure-lightbox-title");
    dialog.setAttribute("aria-describedby", "figure-lightbox-hint");
    const header = element("header", "figure-lightbox-header");
    title = element("h2", "figure-lightbox-title");
    title.id = "figure-lightbox-title";
    const tools = element("div", "figure-lightbox-tools");
    fitButton = button("Fit", "Fit figure to screen", () => setZoom(1));
    fitButton.setAttribute("aria-pressed", "true");
    zoomOutButton = button("−", "Zoom out", () => setZoom(zoom / 1.25));
    zoomLevel = element("output", "figure-lightbox-zoom-level", "100%");
    zoomLevel.setAttribute("aria-label", "Zoom level relative to fit");
    zoomLevel.setAttribute("aria-live", "polite");
    zoomInButton = button("+", "Zoom in", () => setZoom(zoom * 1.25));
    closeButton = button("Close", "Close figure viewer", close);
    closeButton.className = "figure-lightbox-close";
    const escapeHint = element("span", "", "Esc");
    escapeHint.setAttribute("aria-hidden", "true");
    closeButton.append(escapeHint);
    tools.append(fitButton, zoomOutButton, zoomLevel, zoomInButton, closeButton);
    header.append(title, tools);

    viewport = element("div", "figure-lightbox-viewport");
    viewport.tabIndex = 0;
    viewport.setAttribute("role", "region");
    viewport.setAttribute("aria-label", "Scrollable enlarged figure");
    canvas = element("div", "figure-lightbox-canvas");
    status = element("p", "figure-lightbox-status", "Loading figure…");
    status.setAttribute("role", "status");
    viewport.append(status, canvas);
    const hint = element("p", "figure-lightbox-hint", "Zoom for details, then scroll or swipe to move around. Close to return to the page.");
    hint.id = "figure-lightbox-hint";
    dialog.append(header, viewport, hint);
    document.body.append(dialog);

    dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      close();
    });
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) close();
    });
    dialog.addEventListener("keydown", (event) => {
      if (event.key !== "Tab") return;
      const focusable = [...dialog.querySelectorAll("button:not(:disabled), a[href], [tabindex='0']")];
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    });
    dialog.addEventListener("close", () => {
      window.cancelAnimationFrame(frame);
      image?.remove();
      image = null;
      unlockScroll();
      if (trigger?.isConnected) trigger.focus({ preventScroll: true });
      trigger = null;
    });
    window.addEventListener("resize", () => updateImage(true));
  }

  function open(link) {
    if (!dialog) createDialog();
    if (dialog.open) return;
    trigger = link;
    title.textContent = link.dataset.figureTitle || "Figure";
    status.textContent = "Loading figure…";
    status.hidden = false;
    zoom = 1;
    zoomLevel.textContent = "100%";
    fitButton.setAttribute("aria-pressed", "true");
    zoomOutButton.disabled = true;
    zoomInButton.disabled = true;
    image = element("img", "figure-lightbox-image");
    const currentImage = image;
    const inlineImage = link.closest(".figure-viewer-wrap")?.querySelector("img")
      || link.closest(".figure-viewer-action")?.previousElementSibling?.querySelector("img");
    image.alt = inlineImage?.alt || title.textContent;
    image.hidden = true;
    image.draggable = false;
    image.addEventListener("load", () => {
      if (!dialog.open || image !== currentImage) return;
      status.hidden = true;
      image.hidden = false;
      updateImage(true);
    }, { once: true });
    image.addEventListener("error", () => {
      if (!dialog.open || image !== currentImage) return;
      status.textContent = "This figure could not be loaded.";
      const original = element("a", "", "Open original image");
      original.href = link.href;
      original.target = "_blank";
      original.rel = "noopener noreferrer";
      status.append(original);
    }, { once: true });
    canvas.append(image);
    lockScroll();
    dialog.showModal();
    closeButton.focus({ preventScroll: true });
    image.src = link.href;
  }

  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target instanceof Element ? event.target.closest("a[data-figure-zoom]") : null;
    if (!link || typeof HTMLDialogElement === "undefined" || typeof HTMLDialogElement.prototype.showModal !== "function") return;
    event.preventDefault();
    open(link);
  });
})();
