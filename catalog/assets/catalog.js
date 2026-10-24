// Gaudi Cookbook recipe catalog: theme, search, filters and the command builder. No dependencies.
(function () {
  "use strict";

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { /* storage blocked: the page works without it */ }
    return null;
  }

  // Theme
  var toggle = $(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var dark = document.documentElement.classList.toggle("dark");
      store("gc-theme", dark ? "dark" : "light");
    });
  }

  // Copy with a short "Copied" state on the button
  function copy(text, btn) {
    function done() {
      if (!btn) return;
      var label = btn.querySelector("span");
      var old = label ? label.textContent : null;
      btn.classList.add("copied");
      if (label) label.textContent = "Copied";
      setTimeout(function () {
        btn.classList.remove("copied");
        if (label) label.textContent = old;
      }, 1400);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { fallback(text); done(); });
    } else {
      fallback(text);
      done();
    }
  }
  function fallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing else to try */ }
    document.body.removeChild(ta);
  }
  $$("[data-copy-text]").forEach(function (btn) {
    btn.addEventListener("click", function () { copy(btn.getAttribute("data-copy-text"), btn); });
  });

  // Header search
  var search = $(".search");
  if (search) {
    var input = $("input", search);
    var box = $(".search-results", search);
    var prefix = search.getAttribute("data-prefix") || "";
    var index = window.GC_INDEX || [];
    var active = -1;
    var render = function () {
      var q = input.value.trim().toLowerCase();
      if (!q) { box.hidden = true; return; }
      var words = q.split(/\s+/);
      var hits = index.filter(function (r) {
        var hay = (r.id + " " + r.provider + " " + r.desc + " " + r.tasks.join(" ") + " " + r.precision).toLowerCase();
        return words.every(function (w) { return hay.indexOf(w) !== -1; });
      }).slice(0, 8);
      active = hits.length ? 0 : -1;
      box.innerHTML = hits.length
        ? hits.map(function (r, i) {
            return '<a href="' + prefix + r.org + "/" + r.repo + '/"' + (i === 0 ? ' class="active"' : "") + ">" +
              '<div class="r-id">' + esc(r.id) + '</div><div class="r-desc">' + esc(r.desc) + "</div></a>";
          }).join("")
        : '<div class="r-empty">No recipe matches.</div>';
      box.hidden = false;
    };
    input.addEventListener("input", render);
    input.addEventListener("focus", render);
    input.addEventListener("keydown", function (e) {
      var links = $$("a", box);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        if (!links.length) return;
        e.preventDefault();
        active = (active + (e.key === "ArrowDown" ? 1 : -1) + links.length) % links.length;
        links.forEach(function (a, i) { a.classList.toggle("active", i === active); });
      } else if (e.key === "Enter" && links[active]) {
        window.location.href = links[active].getAttribute("href");
      } else if (e.key === "Escape") {
        box.hidden = true;
        input.blur();
      }
    });
    document.addEventListener("click", function (e) { if (!search.contains(e.target)) box.hidden = true; });
    document.addEventListener("keydown", function (e) {
      var tag = (document.activeElement && document.activeElement.tagName) || "";
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") {
        e.preventDefault();
        input.focus();
      }
    });
  }

  // Home filters
  var grid = $("[data-grid]");
  if (grid) {
    var cards = $$(".recipe-card", grid);
    var text = $("[data-filter-text]");
    var count = $("[data-count]");
    var clear = $("[data-clear]");
    var empty = $("[data-empty]");
    var chosen = {};
    var apply = function () {
      var q = (text.value || "").trim().toLowerCase();
      var shown = 0;
      cards.forEach(function (c) {
        var ok = Object.keys(chosen).every(function (k) {
          var vals = chosen[k];
          if (!vals.length) return true;
          var have = (c.getAttribute("data-" + k) || "").split(" ");
          return vals.some(function (v) { return have.indexOf(v) !== -1; });
        });
        if (ok && q) {
          var hay = c.getAttribute("data-text") || "";
          ok = q.split(/\s+/).every(function (w) { return hay.indexOf(w) !== -1; });
        }
        c.hidden = !ok;
        if (ok) shown += 1;
      });
      var any = q || Object.keys(chosen).some(function (k) { return chosen[k].length; });
      count.textContent = shown + " shown";
      clear.hidden = !any;
      empty.hidden = shown !== 0;
    };
    $$("[data-filter]").forEach(function (pill) {
      pill.addEventListener("click", function () {
        var k = pill.getAttribute("data-filter");
        var v = pill.getAttribute("data-value");
        var list = chosen[k] || (chosen[k] = []);
        var at = list.indexOf(v);
        if (at === -1) list.push(v); else list.splice(at, 1);
        pill.classList.toggle("active", at === -1);
        apply();
      });
    });
    text.addEventListener("input", apply);
    clear.addEventListener("click", function () {
      chosen = {};
      text.value = "";
      $$("[data-filter]").forEach(function (p) { p.classList.remove("active"); });
      apply();
    });
  }

  // Command builder
  function highlight(code) {
    return code.split("\n").map(function (line) {
      if (/^\s*#/.test(line)) return '<span class="tok-comment">' + esc(line) + "</span>";
      return esc(line)
        .replace(/(&#39;[^]*?&#39;|&quot;[^]*?&quot;)/g, '<span class="tok-str">$1</span>')
        .replace(/(^|\s)(--?[A-Za-z][\w-]*)/g, '$1<span class="tok-flag">$2</span>')
        .replace(/(^\s*)(export|python3|vllm serve|exec vllm serve|docker run|cat|set|mkdir|containers:)(?=\s|$)/,
          '$1<span class="tok-kw">$2</span>')
        .replace(/(\$[A-Z_][A-Z0-9_]*|\$\{[^}]+\})/g, '<span class="tok-var">$1</span>');
    }).join("\n");
  }

  $$("[data-builder]").forEach(function (root) {
    var payload = JSON.parse($("[data-builder-payload]", root).textContent);
    var features = {};
    payload.features.forEach(function (f) { features[f] = true; });
    var target = store("gc-target") || "shell";
    if (!/^(shell|docker|kubernetes)$/.test(target)) target = "shell";
    var codeEl = $("[data-command]", root);
    var noteEl = $("[data-note]", root);
    var labelEl = $("[data-target-label]", root);
    var labels = { shell: "Shell", docker: "Docker", kubernetes: "Kubernetes" };

    var current = function () {
      var key = payload.features.filter(function (f) { return features[f]; }).sort().join(",");
      return payload.commands[key][target];
    };
    var render = function () {
      codeEl.innerHTML = highlight(current());
      noteEl.innerHTML = payload.notes[target];
      labelEl.textContent = labels[target];
      $$("[data-target]", root).forEach(function (b) { b.classList.toggle("active", b.getAttribute("data-target") === target); });
      $$("[data-feature]", root).forEach(function (b) {
        var on = !!features[b.getAttribute("data-feature")];
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
    };
    $$("[data-feature]", root).forEach(function (b) {
      b.addEventListener("click", function () {
        var f = b.getAttribute("data-feature");
        features[f] = !features[f];
        render();
      });
    });
    $$("[data-target]", root).forEach(function (b) {
      b.addEventListener("click", function () {
        target = b.getAttribute("data-target");
        store("gc-target", target);
        render();
      });
    });
    $("[data-copy-command]", root).addEventListener("click", function (e) { copy(current(), e.currentTarget); });
    $$("[data-popover]", root).forEach(function (b) {
      var body = $('[data-popover-body="' + b.getAttribute("data-popover") + '"]', root);
      b.addEventListener("click", function (e) {
        e.stopPropagation();
        var open = body.hidden;
        $$(".popover", root).forEach(function (p) { p.hidden = true; });
        body.hidden = !open;
      });
      body.addEventListener("click", function (e) { e.stopPropagation(); });
    });
    document.addEventListener("click", function () { $$(".popover", root).forEach(function (p) { p.hidden = true; }); });
    render();
  });
})();
