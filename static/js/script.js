/**
 * script.js
 * GUB Academic AI — Frontend interactions
 */

// ── Sidebar toggle ────────────────────────────────────────────────────
const sidebar   = document.getElementById('sidebar');
const overlay   = document.getElementById('overlay');
const menuBtn   = document.getElementById('menuBtn');
const closeBtn  = document.getElementById('sidebarToggle');

function openSidebar() {
  sidebar.classList.add('open');
  overlay.classList.add('show');
}
function closeSidebar() {
  sidebar.classList.remove('open');
  overlay.classList.remove('show');
}
if (menuBtn)  menuBtn.addEventListener('click', openSidebar);
if (closeBtn) closeBtn.addEventListener('click', closeSidebar);
if (overlay)  overlay.addEventListener('click', closeSidebar);

// ── Animated counters ─────────────────────────────────────────────────
function animateCounter(el) {
  const target = parseInt(el.dataset.target, 10);
  if (isNaN(target)) return;
  const duration = 1200;
  const step     = Math.ceil(target / (duration / 16));
  let current    = 0;
  const timer = setInterval(() => {
    current += step;
    if (current >= target) { current = target; clearInterval(timer); }
    el.textContent = current.toLocaleString();
  }, 16);
}

// Observe and trigger when visible
const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      animateCounter(e.target);
      counterObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.counter').forEach(el => counterObserver.observe(el));

// ── Animate-in elements ───────────────────────────────────────────────
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.animationPlayState = 'running';
      fadeObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.08 });

document.querySelectorAll('.animate-in').forEach(el => {
  el.style.animationPlayState = 'paused';
  fadeObserver.observe(el);
});

// ── Active nav highlight (already done in Jinja, this is fallback) ────
document.querySelectorAll('.nav-link').forEach(link => {
  if (link.href === window.location.href) link.classList.add('active');
});

// ── Tooltip-like hover info for data-table rows ───────────────────────
document.querySelectorAll('.data-table tbody tr').forEach(row => {
  row.addEventListener('mouseenter', () => row.style.transition = 'background 0.2s');
});

// ── Plot image lightbox (simple) ─────────────────────────────────────
document.querySelectorAll('.plot-item img').forEach(img => {
  img.style.cursor = 'zoom-in';
  img.addEventListener('click', () => {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position:fixed;inset:0;background:rgba(0,0,0,0.88);
      display:flex;align-items:center;justify-content:center;
      z-index:9999;cursor:zoom-out;padding:20px;
    `;
    const clone = img.cloneNode();
    clone.style.cssText = 'max-width:90vw;max-height:88vh;border-radius:12px;box-shadow:0 8px 40px rgba(0,0,0,0.7);';
    overlay.appendChild(clone);
    overlay.addEventListener('click', () => overlay.remove());
    document.body.appendChild(overlay);
  });
});

// ── Form validation feedback ──────────────────────────────────────────
const predForm = document.getElementById('predictionForm');
if (predForm) {
  predForm.querySelectorAll('input[required], select[required]').forEach(input => {
    input.addEventListener('blur', () => {
      if (!input.value) {
        input.style.borderColor = '#EF4444';
      } else {
        input.style.borderColor = '';
      }
    });
    input.addEventListener('input', () => {
      if (input.value) input.style.borderColor = '';
    });
  });
}

// ── Smooth scroll for anchor links ────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

// ── Auto-dismiss alert messages after 5s ─────────────────────────────
document.querySelectorAll('.alert').forEach(alert => {
  setTimeout(() => {
    alert.style.transition = 'opacity 0.5s';
    alert.style.opacity = '0';
    setTimeout(() => alert.remove(), 500);
  }, 5000);
});

console.log('🎓 GUB Academic AI — Frontend Loaded');
