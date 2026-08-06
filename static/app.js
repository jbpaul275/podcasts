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
      fetch("/upload", { method: "POST", body, redirect: "follow" })
        .then((r) => { window.location = r.url || "/"; })
        .catch(() => alert("Upload failed."));
    });
  }

  // ---- health line ----
  const health = document.getElementById("health");
  function pollHealth() {
    fetch("/health")
      .then((r) => r.json())
      .then((h) => {
        const bits = [h.worker_alive ? "worker up" : "worker down"];
        if (h.queue_depth) bits.push(h.queue_depth + " queued");
        if (h.worker_current) bits.push("working");
        health.textContent = bits.join(" · ");
      })
      .catch(() => { health.textContent = "server unreachable"; });
  }
  if (health) { pollHealth(); setInterval(pollHealth, 5000); }

  // ---- auto-refresh while anything is mid-pipeline ----
  const working = [...document.querySelectorAll("[data-status]")].some((el) =>
    WORKING.includes(el.dataset.status)
  );
  if (working) setTimeout(() => window.location.reload(), 6000);

  // ---- delete ----
  const del = document.getElementById("delete-episode");
  if (del) {
    del.addEventListener("click", () => {
      if (!confirm("Delete this episode, its PDF, and its audio?")) return;
      fetch("/episode/" + del.dataset.episode, { method: "DELETE" })
        .then((r) => { if (!r.ok) throw new Error(); window.location = "/"; })
        .catch(() => alert("Delete failed."));
    });
  }
})();
