/**
 * Boot screen with real progress, and the unsupported-hardware fallback.
 *
 * The progress bar reflects actual completed init steps. A fake animated bar
 * would be easier and is exactly the kind of self-deception this project is
 * supposed to avoid.
 */

export interface LoadingStep {
  label: string;
  /** Relative cost, used to weight the bar. */
  weight: number;
}

export class LoadingScreen {
  private root: HTMLDivElement;
  private bar: HTMLDivElement;
  private label: HTMLDivElement;
  private detail: HTMLDivElement;

  private steps: LoadingStep[] = [];
  private completed = 0;
  private totalWeight = 0;

  constructor() {
    // The fallback path re-enters boot(), which constructs a second
    // LoadingScreen. Without this, the first one stays mounted forever on top
    // of the scene — an opaque full-screen overlay, i.e. a black screen.
    for (const stale of document.querySelectorAll('.aether-loading')) stale.remove();

    this.root = document.createElement('div');
    this.root.className = 'aether-loading';

    const inner = document.createElement('div');
    inner.className = 'aether-loading-inner';

    const title = document.createElement('h1');
    title.className = 'aether-loading-title';
    title.textContent = 'AETHER';

    const sub = document.createElement('p');
    sub.className = 'aether-loading-sub';
    sub.textContent = 'procedural world · webgpu';

    this.label = document.createElement('div');
    this.label.className = 'aether-loading-label';
    this.label.textContent = 'starting…';

    const track = document.createElement('div');
    track.className = 'aether-loading-track';
    this.bar = document.createElement('div');
    this.bar.className = 'aether-loading-bar';
    track.appendChild(this.bar);

    this.detail = document.createElement('div');
    this.detail.className = 'aether-loading-detail';

    inner.append(title, sub, track, this.label, this.detail);
    this.root.appendChild(inner);
    document.body.appendChild(this.root);
  }

  plan(steps: LoadingStep[]): void {
    this.steps = steps;
    this.totalWeight = steps.reduce((a, s) => a + s.weight, 0);
    this.completed = 0;
    this.render();
  }

  /** Mark the step with this label complete and move to the next. */
  step(label: string): void {
    const idx = this.steps.findIndex((s) => s.label === label);
    if (idx >= 0) {
      this.completed = this.steps.slice(0, idx + 1).reduce((a, s) => a + s.weight, 0);
    }
    this.label.textContent = label;
    this.render();
  }

  setDetail(text: string): void {
    this.detail.textContent = text;
  }

  private render(): void {
    const pct = this.totalWeight > 0 ? (this.completed / this.totalWeight) * 100 : 0;
    this.bar.style.transform = `scaleX(${(pct / 100).toFixed(4)})`;
  }

  async hide(): Promise<void> {
    this.bar.style.transform = 'scaleX(1)';
    this.root.classList.add('is-hidden');
    await new Promise((r) => setTimeout(r, 420));
    this.root.remove();
  }

  /** Replace the loader with an explanation and a degraded-mode button. */
  showUnsupported(reason: string, onFallback: () => void): void {
    this.root.innerHTML = '';
    const inner = document.createElement('div');
    inner.className = 'aether-loading-inner';

    const title = document.createElement('h1');
    title.className = 'aether-loading-title';
    title.textContent = 'WebGPU недоступен';

    const p = document.createElement('p');
    p.className = 'aether-loading-sub';
    p.textContent = reason;

    const p2 = document.createElement('p');
    p2.className = 'aether-loading-detail';
    p2.textContent =
      'Полная версия использует compute-шейдеры: объёмные облака, FFT-океан и GPU-трава ' +
      'без них недоступны. Упрощённый режим работает на WebGL2 с урезанным качеством.';

    const btn = document.createElement('button');
    btn.className = 'aether-btn';
    btn.textContent = 'Запустить в упрощённом режиме';
    btn.addEventListener('click', () => {
      btn.disabled = true;
      btn.textContent = 'Загрузка…';
      onFallback();
    });

    inner.append(title, p, p2, btn);
    this.root.appendChild(inner);
  }

  showFatal(message: string): void {
    this.root.innerHTML = '';
    const inner = document.createElement('div');
    inner.className = 'aether-loading-inner';
    const title = document.createElement('h1');
    title.className = 'aether-loading-title';
    title.textContent = 'Не удалось запустить';
    const p = document.createElement('pre');
    p.className = 'aether-loading-detail';
    p.style.whiteSpace = 'pre-wrap';
    p.style.textAlign = 'left';
    p.textContent = message;
    inner.append(title, p);
    this.root.appendChild(inner);
  }
}
