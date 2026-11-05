import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../public/figure-viewer.js", import.meta.url), "utf8");

function browserHarness() {
  let document;
  class Node {
    constructor(tag = "div") {
      this.tagName = tag;
      this.className = "";
      this.children = [];
      this.attributes = {};
      this.dataset = {};
      this.style = { overflow: "", position: "", top: "", width: "", scrollBehavior: "" };
      this.listeners = new Map();
      this.clientWidth = 900;
      this.clientHeight = 500;
      this.scrollWidth = 1800;
      this.scrollHeight = 1000;
      this.isConnected = true;
    }
    addEventListener(name, callback) {
      const listeners = this.listeners.get(name) ?? [];
      listeners.push(callback);
      this.listeners.set(name, listeners);
    }
    emit(name, event = {}) {
      event.target ??= this;
      event.preventDefault ??= () => { event.defaultPrevented = true; };
      for (const callback of this.listeners.get(name) ?? []) callback(event);
      return event;
    }
    setAttribute(key, value) { this.attributes[key] = value; }
    append(...nodes) {
      for (const node of nodes) { node.parentElement = this; this.children.push(node); }
    }
    remove() {
      if (this.parentElement) this.parentElement.children = this.parentElement.children.filter((node) => node !== this);
      this.isConnected = false;
    }
    focus() { document.activeElement = this; }
    contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
    closest(selector) {
      if (selector === "a[data-figure-zoom]" && this.tagName === "a" && "figureZoom" in this.dataset) return this;
      if (selector.startsWith(".") && this.className.split(" ").includes(selector.slice(1))) return this;
      return this.parentElement?.closest(selector) ?? null;
    }
    querySelectorAll(selector) {
      const all = this.children.flatMap((child) => [child, ...child.querySelectorAll("*")]);
      if (selector === "*") return all;
      if (selector === "img") return all.filter((child) => child.tagName === "img");
      return all.filter((child) => child.tagName === "button" && !child.disabled || child.tagName === "a" && child.href || child.tabIndex === 0);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
    showModal() { this.open = true; }
    close() { this.open = false; this.emit("close"); }
  }
  document = new Node("document");
  document.body = new Node("body");
  document.documentElement = new Node("html");
  document.createElement = (tag) => new Node(tag);
  const window = new Node("window");
  window.scrollX = 0;
  window.scrollY = 750;
  window.scrollTo = (x, y) => { window.restoredScroll = [x, y]; };
  window.requestAnimationFrame = (callback) => { callback(); return 1; };
  window.cancelAnimationFrame = () => {};
  const context = vm.createContext({ window, document, Element: Node, HTMLDialogElement: Node });
  const trigger = new Node("a");
  trigger.dataset = { figureZoom: "", figureTitle: "System overview" };
  trigger.href = "https://example.com/Robotic-0/assets/main-overview.svg?v=latest";
  const load = () => vm.runInContext(source, context);
  const click = (options = {}) => document.emit("click", { target: trigger, button: 0, ...options });
  const dialog = () => document.body.children.find((node) => node.tagName === "dialog");
  const find = (className) => dialog().querySelectorAll("*").find((node) => node.className === className);
  const labelledButton = (name) => dialog().querySelectorAll("*").find((node) => node.attributes["aria-label"] === name);
  return { load, click, dialog, find, labelledButton, trigger, document, window };
}

test("the shared figure controller is SSR-safe and installs only one delegated handler", () => {
  assert.doesNotThrow(() => vm.runInNewContext(source, {}));
  const harness = browserHarness();
  harness.load();
  harness.load();
  assert.equal(harness.document.listeners.get("click").length, 1);
  assert.equal(harness.click({ metaKey: true }).defaultPrevented, undefined);
  assert.equal(harness.dialog(), undefined);
});

test("the viewer preserves scoped SVG URLs, supports zoom, traps focus, and restores the page on Escape", () => {
  const harness = browserHarness();
  harness.load();
  const event = harness.click();
  assert.equal(event.defaultPrevented, true);
  assert.equal(harness.dialog().open, true);
  assert.equal(harness.document.body.style.position, "fixed");
  assert.equal(harness.document.body.style.top, "-750px");
  assert.equal(harness.document.activeElement, harness.labelledButton("Close figure viewer"));
  const image = harness.find("figure-lightbox-image");
  assert.equal(image.src, harness.trigger.href);
  image.naturalWidth = 1763;
  image.naturalHeight = 1170;
  image.emit("load");
  assert.equal(image.hidden, false);
  assert.equal(harness.find("figure-lightbox-status").hidden, true);
  const fitWidth = parseFloat(image.style.width);
  harness.labelledButton("Zoom in").emit("click");
  assert.equal(harness.find("figure-lightbox-zoom-level").textContent, "125%");
  assert.ok(parseFloat(image.style.width) > fitWidth);
  harness.labelledButton("Fit figure to screen").emit("click");
  assert.equal(harness.find("figure-lightbox-zoom-level").textContent, "100%");
  harness.find("figure-lightbox-viewport").focus();
  assert.equal(harness.dialog().emit("keydown", { key: "Tab" }).defaultPrevented, true);
  assert.equal(harness.document.activeElement, harness.labelledButton("Fit figure to screen"));
  harness.dialog().emit("keydown", { key: "Tab", shiftKey: true });
  assert.equal(harness.document.activeElement, harness.find("figure-lightbox-viewport"));
  assert.equal(harness.dialog().emit("cancel").defaultPrevented, true);
  assert.equal(harness.dialog().open, false);
  assert.equal(harness.document.body.style.position, "");
  assert.equal(harness.document.body.style.overflow, "");
  assert.equal(harness.document.documentElement.style.overflow, "");
  assert.deepEqual(harness.window.restoredScroll, [0, 750]);
  assert.equal(harness.document.activeElement, harness.trigger);
  assert.equal(harness.find("figure-lightbox-image"), undefined);
});

test("load failures retain an original-image link and closing remains available", () => {
  const harness = browserHarness();
  harness.load();
  harness.click();
  harness.find("figure-lightbox-image").emit("error");
  const status = harness.find("figure-lightbox-status");
  assert.match(status.textContent, /could not be loaded/);
  assert.equal(status.children[0].href, harness.trigger.href);
  harness.labelledButton("Close figure viewer").emit("click");
  assert.equal(harness.dialog().open, false);
  assert.equal(harness.document.body.style.position, "");
});
