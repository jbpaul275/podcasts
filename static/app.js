// Vanilla JS, no build step. Drag-drop upload, /health poll, status auto-refresh.

(function () {
  const WORKING = ["queued", "extracting", "scripting", "synthesizing", "assembling"];

  // ---- drag and drop upload ----
  const zone = document.getElementById("dropzone");
  if (zone) {
    const input = document.getElementById("fileinput");
    document.getElementById("pickfile").addEventListener("click", () => input.click());
    input.addEventListener("change", () => { if (input.files.length) zone.submit(); });

    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); })
    );
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("over"); })
    );
    zone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".pdf")) { alert("PDFs only."); return; }
      const body = new FormData();
      body.append("file", file);
      zone.classList.add("busy");
      fetch("/upload", { method: "POST", body, redirect: "follow" })
        .then((r) => {
          // The server redirects to the episode on success, to the duplicate's
          // episode, or back to /admin with a readable error. Following it is
          // always the right move; a non-OK response means something else broke.
          if (!r.ok) throw new Error(r.status === 401 ? "Not signed in." : "Upload failed.");
          window.location = r.url || "/admin";
        })
        .catch((e) => {
          zone.classList.remove("busy");
          alert(e.message || "Upload failed.");
        });
    });
  }

  // ---- inline play buttons on the library ----
  // One shared audio element, so starting a second episode stops the first.
  const mini = document.getElementById("miniplayer");
  if (mini) {
    let active = null;

    function reset(btn) {
      if (!btn) return;
      btn.classList.remove("playing");
      btn.querySelector(".glyph").textContent = "▶";
      btn.setAttribute("aria-label", btn.dataset.label);
    }

    document.querySelectorAll(".play").forEach((btn) => {
      btn.dataset.label = btn.getAttribute("aria-label");
      btn.addEventListener("click", () => {
        if (active === btn) {
          if (mini.paused) { mini.play(); } else { mini.pause(); }
          return;
        }
        reset(active);
        active = btn;
        mini.src = btn.dataset.src;
        mini.play().catch(() => {
          reset(btn);
          active = null;
          alert("Could not play this episode.");
        });
      });
    });

    mini.addEventListener("play", () => {
      if (!active) return;
      active.classList.add("playing");
      active.querySelector(".glyph").textContent = "❚❚";
      active.setAttribute("aria-label", "Pause");
    });
    mini.addEventListener("pause", () => {
      if (!active) return;
      active.classList.remove("playing");
      active.querySelector(".glyph").textContent = "▶";
    });
    mini.addEventListener("ended", () => { reset(active); active = null; });
  }

  // ---- health line ----
  // Silent when nothing needs saying. A permanent "worker up" is noise; the
  // useful signals are that the worker died, or that work is in flight.
  const health = document.getElementById("health");
  function pollHealth() {
    fetch("/health")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((h) => {
        const bits = [];
        if (!h.worker_alive) bits.push("worker down");
        if (h.worker_current) bits.push("processing");
        if (h.queue_depth) bits.push(h.queue_depth + " queued");
        health.textContent = bits.join(" · ");
        health.classList.toggle("bad", !h.worker_alive);
      })
      .catch(() => {
        health.textContent = "server unreachable";
        health.classList.add("bad");
      });
  }
  if (health) { pollHealth(); setInterval(pollHealth, 5000); }

  // ---- auto-refresh while anything is mid-pipeline ----
  const queueEl = document.getElementById("queue");
  const queued = queueEl ? Number(queueEl.dataset.count) > 0 : false;
  const working = queued || [...document.querySelectorAll("[data-status]")].some((el) =>
    WORKING.includes(el.dataset.status)
  );
  if (working) setTimeout(() => window.location.reload(), 6000);

  // ---- delete (episode page, and each row in the failures list) ----
  document.querySelectorAll(".delete-episode").forEach((del) => {
    del.addEventListener("click", () => {
      if (!confirm("Delete this episode, its PDF, and its audio?")) return;
      fetch("/episode/" + del.dataset.episode, { method: "DELETE" })
        .then((r) => { if (!r.ok) throw new Error(); window.location.reload(); })
        .catch(() => alert("Delete failed."));
    });
  });
})();
