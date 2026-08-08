/**
 * GUB Academic AI — Interactive Frontend Script
 * Modern UX interactions: smooth counter animation, form synchronization,
 * mobile drawer management, and loading state handlers.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── Mobile Sidebar Drawer Toggle ──────────────────────────────────────────
  const sidebar = document.getElementById('sidebar');
  const mobileToggle = document.getElementById('mobileNavToggle');
  const sidebarClose = document.getElementById('sidebarCloseBtn');

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.add('mobile-open');
    });
  }

  if (sidebarClose && sidebar) {
    sidebarClose.addEventListener('click', () => {
      sidebar.classList.remove('mobile-open');
    });
  }

  // Close mobile sidebar on clicking outside
  document.addEventListener('click', (e) => {
    if (sidebar && sidebar.classList.contains('mobile-open')) {
      if (!sidebar.contains(e.target) && mobileToggle && !mobileToggle.contains(e.target)) {
        sidebar.classList.remove('mobile-open');
      }
    }
  });

  // ── Animated Counter Numbers on Page Load ───────────────────────────────
  const counterElements = document.querySelectorAll('.animate-counter');

  const startCounters = () => {
    counterElements.forEach(el => {
      const target = parseFloat(el.getAttribute('data-target'));
      if (isNaN(target)) return;

      const isFloat = el.getAttribute('data-float') === 'true';
      const duration = 1200; // ms
      const steps = 40;
      const stepTime = duration / steps;
      const increment = target / steps;
      let current = 0;

      const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
          current = target;
          clearInterval(timer);
        }
        el.textContent = isFloat ? current.toFixed(4) : Math.round(current).toLocaleString();
      }, stepTime);
    });
  };

  if (counterElements.length > 0) {
    startCounters();
  }

  // ── Synchronized Dual-Control Range Sliders & Inputs ──────────────────────
  const setupRangeSync = (inputId, sliderId, badgeId) => {
    const input = document.getElementById(inputId);
    const slider = document.getElementById(sliderId);
    const badge = document.getElementById(badgeId);

    if (!input || !slider) return;

    const updateValue = (val) => {
      input.value = val;
      slider.value = val;
      if (badge) {
        badge.textContent = inputId.includes('pass_rate') ? `${val}%` : val;
      }
    };

    input.addEventListener('input', (e) => updateValue(e.target.value));
    slider.addEventListener('input', (e) => updateValue(e.target.value));
  };

  setupRangeSync('previous_enrollment', 'enrollment_slider', 'enrollment_badge');
  setupRangeSync('previous_pass_rate', 'pass_rate_slider', 'pass_rate_badge');
  setupRangeSync('previous_resource_usage', 'resource_slider', 'resource_badge');

  // ── Segmented Toggle Buttons (Lab Facility Required) ────────────────────
  const labRequiredInput = document.getElementById('lab_required');
  const toggleOptions = document.querySelectorAll('.lab-toggle-option');
  const labCapacityGroup = document.getElementById('lab_capacity_group');

  if (labRequiredInput && toggleOptions.length > 0) {
    toggleOptions.forEach(opt => {
      opt.addEventListener('click', () => {
        toggleOptions.forEach(o => o.classList.remove('active'));
        opt.classList.add('active');
        const val = opt.getAttribute('data-value');
        labRequiredInput.value = val;

        if (labCapacityGroup) {
          labCapacityGroup.style.display = (val === '1') ? 'flex' : 'none';
        }
      });
    });
  }

  // ── Form Submit Loading Overlay State ─────────────────────────────────────
  const predictionForm = document.getElementById('predictionForm');
  const submitBtn = document.getElementById('submitPredictionBtn');

  if (predictionForm && submitBtn) {
    predictionForm.addEventListener('submit', () => {
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<i class="bi bi-arrow-repeat spin-icon"></i> Analyzing Academic Patterns...';
      submitBtn.style.opacity = '0.85';
      submitBtn.style.cursor = 'wait';
    });
  }

});
