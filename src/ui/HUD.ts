/**
 * Minimal diegetic HUD: a crosshair, a transient hint line, and the help sheet.
 * Deliberately sparse — this is a world to walk through, not a game UI.
 */

const HOTKEYS: ReadonlyArray<readonly [string, string]> = [
  ['W A S D', 'движение'],
  ['Shift', 'бег / ускорение'],
  ['Alt', 'шаг / медленно'],
  ['Space', 'прыжок / вверх'],
  ['Q E', 'вниз / вверх (fly)'],
  ['Мышь', 'обзор (клик — захват курсора)'],
  ['Колесо', 'скорость fly-камеры'],
  ['F1', 'fly-камера / игрок'],
  ['F2', 'скриншот без HUD'],
  ['F3', 'статистика'],
  ['F4', 'настройки'],
  ['P', 'фоторежим'],
  ['H', 'эта справка'],
  ['Esc', 'отпустить курсор'],
];

export class HUD {
  private root: HTMLDivElement;
  private crosshair: HTMLDivElement;
  private hint: HTMLDivElement;
  private help: HTMLDivElement;
  private hintTimer = 0;
  private helpVisible = false;

  constructor() {
    this.root = document.createElement('div');
    this.root.className = 'aether-hud';

    this.crosshair = document.createElement('div');
    this.crosshair.className = 'aether-crosshair';
    this.crosshair.setAttribute('aria-hidden', 'true');

    this.hint = document.createElement('div');
    this.hint.className = 'aether-hint';
    this.hint.setAttribute('role', 'status');

    this.help = document.createElement('div');
    this.help.className = 'aether-help';
    this.help.innerHTML =
      '<h2>Управление</h2><dl>' +
      HOTKEYS.map(([k, v]) => `<dt><kbd>${k}</kbd></dt><dd>${v}</dd>`).join('') +
      '</dl>';
    this.help.style.display = 'none';

    this.root.append(this.crosshair, this.hint, this.help);
    document.body.appendChild(this.root);

    window.addEventListener('keydown', this.onKey);
  }

  private onKey = (e: KeyboardEvent): void => {
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
    if (e.code === 'KeyH') {
      this.helpVisible = !this.helpVisible;
      this.help.style.display = this.helpVisible ? '' : 'none';
    }
  };

  showHint(text: string, ms = 2600): void {
    this.hint.textContent = text;
    this.hint.classList.add('is-visible');
    window.clearTimeout(this.hintTimer);
    this.hintTimer = window.setTimeout(() => {
      this.hint.classList.remove('is-visible');
    }, ms);
  }

  setCrosshairVisible(v: boolean): void {
    this.crosshair.style.opacity = v ? '' : '0';
  }

  /** Hide everything for a clean screenshot. */
  setVisible(v: boolean): void {
    this.root.style.display = v ? '' : 'none';
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKey);
    window.clearTimeout(this.hintTimer);
    this.root.remove();
  }
}
