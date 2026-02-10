/**
 * ClaimWatch AI — Main JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    // ─── Form validation highlight ──────────────
    const form = document.getElementById('claimForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            const inputs = form.querySelectorAll('input[required], select[required]');
            let valid = true;
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    input.style.borderColor = '#ef4444';
                    valid = false;
                } else {
                    input.style.borderColor = '';
                }
            });
            if (!valid) {
                e.preventDefault();
            }
        });
    }

    // ─── Animate gauge on result page ───────────
    const gaugeValue = document.getElementById('gaugeValue');
    if (gaugeValue) {
        const target = parseFloat(gaugeValue.textContent);
        let current = 0;
        const duration = 1200;
        const start = performance.now();

        function animate(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            current = target * ease;
            gaugeValue.textContent = current.toFixed(1) + '%';
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        }
        requestAnimationFrame(animate);
    }

    // ─── Animate feature bars ───────────────────
    const featureBars = document.querySelectorAll('.feature-bar-fill');
    featureBars.forEach(bar => {
        const width = bar.getAttribute('data-width');
        if (width) {
            // Set width via JS to avoid CSS linter errors in templates
            setTimeout(() => {
                bar.style.width = width + '%';
            }, 400);
        }
    });
});
