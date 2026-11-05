"use client";

import { useEffect, useRef, useState } from "react";

const sections = [
  ["overview", "Overview"],
  ["data", "Data"],
  ["method", "Method"],
  ["evaluation", "Evaluation"],
  ["demos", "Demos"],
  ["conclusion", "Conclusion"],
  ["citation", "Citation"],
] as const;

type SectionId = (typeof sections)[number][0];

export default function ProjectNav() {
  const navRef = useRef<HTMLElement>(null);
  const [activeSection, setActiveSection] = useState<SectionId>("overview");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let frame: number | null = null;

    function update() {
      const anchor = Math.min(window.innerHeight * 0.35, 180);
      let nextSection: SectionId = sections[0][0];

      for (const [id] of sections) {
        const section = document.getElementById(id);
        if (section && section.getBoundingClientRect().top <= anchor) {
          nextSection = id;
        }
      }

      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const nextProgress = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;

      if (nextProgress >= 99.9) {
        nextSection = sections[sections.length - 1][0];
      }

      setActiveSection(nextSection);
      setProgress(Math.min(100, Math.max(0, nextProgress)));
    }

    function scheduleUpdate() {
      if (frame !== null) return;
      frame = window.requestAnimationFrame(() => {
        frame = null;
        update();
      });
    }

    update();
    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);

    return () => {
      window.removeEventListener("scroll", scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
      if (frame !== null) window.cancelAnimationFrame(frame);
    };
  }, []);

  useEffect(() => {
    const nav = navRef.current;
    const activeLink = nav?.querySelector<HTMLAnchorElement>(
      'a[aria-current="location"]',
    );
    if (!nav || !activeLink) return;

    const left =
      activeLink.offsetLeft - (nav.clientWidth - activeLink.offsetWidth) / 2;
    nav.scrollTo({
      left: Math.max(0, left),
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  }, [activeSection]);

  return (
    <>
      <nav ref={navRef} aria-label="Project sections">
        {sections.map(([id, label]) => (
          <a
            href={`#${id}`}
            aria-current={activeSection === id ? "location" : undefined}
            key={id}
          >
            {label}
          </a>
        ))}
      </nav>
      <div
        className="reading-progress"
        role="progressbar"
        aria-label="Page reading progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress)}
      >
        <span style={{ width: `${progress}%` }} />
      </div>
    </>
  );
}
