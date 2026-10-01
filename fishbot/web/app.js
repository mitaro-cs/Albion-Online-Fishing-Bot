import { I18N } from './i18n.js';

// ── utilities ───────────────────────────────────────────────────────────────

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : String(kid));
  return el;
}

const ICONS = {
  play: '<path d="M7.5 5.2v13.6L18.5 12z"/>',
  pause: '<path d="M8.5 5.5v13M15.5 5.5v13"/>',
  x: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
  more: '<circle cx="12" cy="5.5" r=".9"/><circle cx="12" cy="12" r=".9"/><circle cx="12" cy="18.5" r=".9"/>',
  menu: '<path d="M5 8h14M5 12h14M5 16h14"/>',
  chevL: '<path d="M14.5 6.5L9 12l5.5 5.5"/>',
  chevR: '<path d="M9.5 6.5L15 12l-5.5 5.5"/>',
  chevD: '<path d="M7 10l5 5 5-5"/>',
  camera: '<path d="M4.5 8.8A2.3 2.3 0 016.8 6.5h1.6l1.5-2h4.2l1.5 2h1.6a2.3 2.3 0 012.3 2.3v7.9a2.3 2.3 0 01-2.3 2.3H6.8a2.3 2.3 0 01-2.3-2.3z"/><circle cx="12" cy="12.6" r="3.3"/>',
  checkCircle: '<circle cx="12" cy="12" r="8.5"/><path d="M8.6 12.3l2.3 2.3 4.6-4.9"/>',
  check: '<path d="M5.5 12.5l4 4 9-9.5"/>',
  spark: '<path d="M12 3.5l1.7 5 5 1.7-5 1.7-1.7 5-1.7-5-5-1.7 5-1.7z"/>',
  plus: '<path d="M12 5.5v13M5.5 12h13"/>',
  minus: '<path d="M5.5 12h13"/>',
  fit: '<path d="M4.5 9V5.5a1 1 0 011-1H9M15 4.5h3.5a1 1 0 011 1V9M19.5 15v3.5a1 1 0 01-1 1H15M9 19.5H5.5a1 1 0 01-1-1V15"/>',
  eye: '<path d="M2.8 12S6.2 5.8 12 5.8 21.2 12 21.2 12 17.8 18.2 12 18.2 2.8 12 2.8 12z"/><circle cx="12" cy="12" r="2.8"/>',
  layout: '<rect x="4.5" y="4.5" width="15" height="15" rx="2.5"/><path d="M4.5 10h15M10 10v9.5"/>',
  history: '<path d="M4 12a8 8 0 102.4-5.7"/><path d="M4 5v4h4"/><path d="M12 8v4.3l2.8 1.8"/>',
  chart: '<path d="M6 18.5v-6M12 18.5v-12M18 18.5v-8.5"/>',
  cube: '<path d="M12 3.8l7.5 4.2v8L12 20.2 4.5 16V8z"/><path d="M12 12l7.5-4M12 12v8.2M12 12L4.5 8"/>',
  folder: '<path d="M4 7.8a2 2 0 012-2h3.6l2 2.4H18a2 2 0 012 2v6.6a2 2 0 01-2 2H6a2 2 0 01-2-2z"/>',
  gear: '<circle cx="12" cy="12" r="2.8"/><path d="M19.2 14.6a1.5 1.5 0 00.3 1.6l.1.1a1.8 1.8 0 11-2.6 2.6l-.1-.1a1.5 1.5 0 00-1.6-.3 1.5 1.5 0 00-.9 1.4v.2a1.8 1.8 0 11-3.6 0v-.1a1.5 1.5 0 00-1-1.4 1.5 1.5 0 00-1.6.3l-.1.1a1.8 1.8 0 11-2.6-2.6l.1-.1a1.5 1.5 0 00.3-1.6 1.5 1.5 0 00-1.4-.9h-.2a1.8 1.8 0 110-3.6h.1a1.5 1.5 0 001.4-1 1.5 1.5 0 00-.3-1.6l-.1-.1a1.8 1.8 0 112.6-2.6l.1.1a1.5 1.5 0 001.6.3h.1a1.5 1.5 0 00.9-1.4v-.2a1.8 1.8 0 113.6 0v.1a1.5 1.5 0 00.9 1.4 1.5 1.5 0 001.6-.3l.1-.1a1.8 1.8 0 112.6 2.6l-.1.1a1.5 1.5 0 00-.3 1.6v.1a1.5 1.5 0 001.4.9h.2a1.8 1.8 0 110 3.6h-.1a1.5 1.5 0 00-1.4.9z"/>',
  expand: '<path d="M14 4.5h5.5V10M10 19.5H4.5V14M19.5 4.5L13 11M4.5 19.5L11 13"/>',
  target: '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.5"/>',
  bar: '<rect x="3.5" y="9" width="17" height="6" rx="3"/><path d="M13 9v6"/>',
  bubbles: '<circle cx="8" cy="9" r="2.3"/><circle cx="15.8" cy="7.6" r="1.6"/><circle cx="13.6" cy="14.6" r="3.2"/>',
  refresh: '<path d="M19.5 11.5a7.5 7.5 0 10-2.2 5.3"/><path d="M19.5 5v6.5H13"/>',
  trash: '<path d="M5.5 7.5h13M10 7.5V5.5h4v2M7.5 7.5l.8 11h7.4l.8-11"/>',
  pin: '<path d="M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0113 0c0 5.4-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.3"/>',
  edit: '<path d="M14.5 5.5l4 4L8 20H4v-4z"/>',
  calendar: '<rect x="4.5" y="5.5" width="15" height="14" rx="2.5"/><path d="M4.5 10h15M9 3.8v3.4M15 3.8v3.4"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5v-2a2 2 0 00-2-2h-7a2 2 0 00-2 2v7a2 2 0 002 2h2"/>',
};

function ic(name) {
  const tpl = document.createElement('template');
  tpl.innerHTML = `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ''}</svg>`;
  return tpl.content.firstChild;
}

const GLYPH = `<svg class="glyph" viewBox="0 0 120 120" fill="none" stroke="rgba(255,255,255,.92)" stroke-width="1.8" stroke-linecap="round">
  <path d="M60 10v28"/><circle cx="60" cy="54" r="16"/><path d="M44 54h32"/>
  <path d="M46 54a14 14 0 0028 0z" fill="rgba(255,255,255,.18)" stroke="none"/>
  <ellipse cx="60" cy="86" rx="30" ry="6" opacity=".6"/><ellipse cx="60" cy="88" rx="48" ry="10" opacity=".3"/></svg>`;

// ── i18n ────────────────────────────────────────────────────────────────────

let lang = 'en';
const UNITS = { en: { ms: 'ms', s: 's', px: 'px', min: 'min', fps: 'fps' }, ru: { ms: 'мс', s: 'с', px: 'px', min: 'мин', fps: 'fps' } };

function t(key, params = {}) {
  let s = I18N[lang]?.[key] ?? I18N.en[key] ?? key;
  for (const [k, v] of Object.entries(params)) s = s.replaceAll(`{${k}}`, v);
  return s;
}

function applyI18n() {
  document.documentElement.lang = lang;
  $$('[data-t]').forEach(el => { el.textContent = t(el.dataset.t); });
  $$('[data-tip]').forEach(el => { el.title = t(el.dataset.tip); });
  $('#lang').textContent = lang === 'ru' ? 'EN' : 'RU';
  $('#lang').title = t('tab.lang');
}

// ── API ─────────────────────────────────────────────────────────────────────

const TOKEN = new URLSearchParams(location.search).get('token') || '';
let failures = 0;

async function call(method, ...args) {
  try {
    const res = await fetch(`/api/${method}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Token': TOKEN },
      body: JSON.stringify({ args }),
    });
    const json = await res.json();
    if (failures >= 3) toast(t('toast.ready'));
    failures = 0;
    document.body.classList.remove('offline');
    return json;
  } catch (e) {
    if (++failures === 3) toast(t('err.offline'), 'error');
    document.body.classList.add('offline');
    return { ok: false, error: 'offline' };
  }
}

function errText(r) {
  const p = r.params || {};
  return t(`err.${r.error}`, { ...p, name: p.name ? t(`name.${p.name}`) : '', message: r.message || '' });
}

async function act(method, ...args) {
  const r = await call(method, ...args);
  if (!r.ok && r.error !== 'offline') toast(errText(r), 'error');
  return r;
}

// ── state ───────────────────────────────────────────────────────────────────

const S = {
  boot: null, cfg: null, schema: null, templates: {}, snap: null, logs: [], lastLog: 0,
  zoom: 1, pan: [0, 0], layout: {}, wires: true, dock: true, panel: null,
  preview: { test: null, live: true },
};

const getPath = path => path.split('.').reduce((o, k) => (o == null ? o : o[k]), S.cfg);
const meta = path => path.split('.').reduce((o, k) => (o == null ? o : o[k]), S.schema) || {};
const binders = new Map();
let scope = null; // collects binders created while rendering a panel, so they can be dropped on re-render

function bind(path, fn) {
  if (!binders.has(path)) binders.set(path, []);
  binders.get(path).push(fn);
  scope?.push([path, fn]);
  fn(path.startsWith('@') ? null : getPath(path));
}

function unbind(list) {
  for (const [path, fn] of list || []) {
    const arr = binders.get(path);
    const i = arr ? arr.indexOf(fn) : -1;
    if (i >= 0) arr.splice(i, 1);
  }
}

function notify(path) {
  for (const fn of binders.get(path) || []) fn(path.startsWith('@') ? null : getPath(path));
}

let pending = {};
let saveTimer = 0;

function setValue(path, value) {
  const keys = path.split('.');
  let o = S.cfg, p = pending;
  for (const k of keys.slice(0, -1)) { o = o[k]; p = p[k] ??= {}; }
  o[keys.at(-1)] = value;
  p[keys.at(-1)] = value;
  notify(path);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 260);
}

async function flush() {
  const patch = pending;
  pending = {};
  if (!Object.keys(patch).length) return;
  const r = await act('update', patch);
  if (r.ok) applyConfig(r.config);
}

function applyConfig(cfg) {
  if (Object.keys(pending).length) return; // newer local edits will be flushed — don't clobber them
  S.cfg = cfg;
  for (const path of binders.keys()) notify(path);
  $('#kbd-toggle').textContent = cfg.system.hotkey_toggle.toUpperCase();
  drawWires();
}

// ── controls ────────────────────────────────────────────────────────────────

function decimals(step) {
  if (!step || step >= 1) return 0;
  return Math.min(3, (String(step).split('.')[1] || '').length);
}

const OFF_AT_ZERO = new Set(['session.max_catches', 'session.max_minutes', 'session.break_every_min']);

function fmt(path, v) {
  const m = meta(path);
  if (OFF_AT_ZERO.has(path) && Number(v) === 0) return t('zero.off');
  const s = Number(v).toFixed(decimals(m.step));
  return m.unit ? `${s} ${UNITS[lang][m.unit] || m.unit}` : s;
}

function coerce(path, raw) {
  const m = meta(path);
  let v = parseFloat(String(raw).replace(',', '.'));
  if (!Number.isFinite(v)) return getPath(path);
  v = clamp(v, m.min ?? -Infinity, m.max ?? Infinity);
  return m.type === 'int' ? Math.round(v) : +v.toFixed(4);
}

function row(labelKey, control, cls = '') {
  return h('div', { class: `row ${cls}` }, h('label', { title: t(labelKey) }, t(labelKey)), control);
}

function select(path, options, parse = v => v) {
  const el = h('select', { class: 'ctl', onchange: e => setValue(path, parse(e.target.value)) },
    options.map(([v, k]) => h('option', { value: v }, t(k))));
  bind(path, v => { el.value = v; });
  return el;
}

function stepper(path) {
  const m = meta(path);
  const step = m.step || 1;
  const input = h('input', { class: 'ctl', inputmode: 'decimal', spellcheck: 'false' });
  const show = () => { input.value = fmt(path, getPath(path)); };
  input.addEventListener('focus', () => { input.value = getPath(path); input.select(); });
  input.addEventListener('blur', () => { setValue(path, coerce(path, input.value)); show(); });
  input.addEventListener('keydown', e => { if (e.key === 'Enter') input.blur(); });
  const nudge = dir => setValue(path, coerce(path, getPath(path) + dir * step));
  const btn = (cls, dir) => {
    const b = h('button', { class: cls, type: 'button', tabindex: '-1' });
    let timer = 0;
    const stopRepeat = () => clearInterval(timer);
    b.addEventListener('pointerdown', e => {
      e.preventDefault();
      nudge(dir);
      let n = 0;
      stopRepeat();
      timer = setInterval(() => { if (++n > 4) nudge(dir); }, 80);
    });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach(ev => b.addEventListener(ev, stopRepeat));
    return b;
  };
  bind(path, () => { if (document.activeElement !== input) show(); });
  return h('div', { class: 'stepper' }, btn('dec', -1), input, btn('inc', 1));
}

function slider(path) {
  const m = meta(path);
  const input = h('input', { type: 'range', min: m.min, max: m.max, step: m.step || 1 });
  const val = h('span', { class: 'val' });
  const paint = v => {
    input.style.setProperty('--p', `${((v - m.min) / (m.max - m.min)) * 100}%`);
    val.textContent = fmt(path, v);
  };
  input.addEventListener('input', () => { paint(+input.value); setValue(path, coerce(path, input.value)); });
  bind(path, v => { input.value = v; paint(v); });
  return h('div', { class: 'slider' }, input, val);
}

function toggle(path) {
  const el = h('button', { class: 'toggle', type: 'button', role: 'switch', onclick: () => setValue(path, !getPath(path)) });
  bind(path, v => { el.classList.toggle('on', !!v); el.setAttribute('aria-checked', String(!!v)); });
  return el;
}

function textInput(path) {
  const el = h('input', { class: 'ctl', spellcheck: 'false' });
  el.addEventListener('change', () => setValue(path, el.value.trim()));
  bind(path, v => { if (document.activeElement !== el) el.value = v; });
  return el;
}

function keyName(e) {
  if (/^F\d{1,2}$/.test(e.key)) return e.key.toLowerCase();
  if (e.key.length === 1 && /[a-z0-9]/i.test(e.key)) return e.key.toLowerCase();
  return { ' ': 'space', Enter: 'enter', Tab: 'tab' }[e.key] || null;
}

function keyInput(path) {
  const el = h('button', { class: 'ctl', type: 'button' });
  let listening = false;
  const show = () => { el.textContent = listening ? t('key.press') : String(getPath(path)).toUpperCase(); };
  el.addEventListener('click', () => {
    listening = true;
    show();
    const onKey = e => {
      e.preventDefault();
      e.stopPropagation();
      window.removeEventListener('keydown', onKey, true);
      listening = false;
      const name = e.key === 'Escape' ? null : keyName(e);
      if (name) setValue(path, name);
      show();
    };
    window.addEventListener('keydown', onKey, true);
  });
  bind(path, show);
  return el;
}

function hsvInput(path) {
  const inputs = [0, 1, 2].map(i => {
    const el = h('input', { class: 'ctl', inputmode: 'numeric' });
    el.addEventListener('change', () => {
      const v = [...getPath(path)];
      v[i] = clamp(Math.round(+el.value || 0), 0, i === 0 ? 179 : 255);
      setValue(path, v);
    });
    return el;
  });
  bind(path, v => inputs.forEach((el, i) => { if (document.activeElement !== el) el.value = v[i]; }));
  return h('div', { class: 'hsv' }, inputs);
}

function fold(...rows) {
  const box = h('div', { class: 'fold' }, h('div', {}, rows));
  const btn = h('button', { class: 'more', type: 'button' }, h('span', {}, t('btn.more')), ic('chevD'));
  btn.addEventListener('click', () => {
    const open = box.classList.toggle('open');
    btn.classList.toggle('open', open);
    btn.firstChild.textContent = t(open ? 'btn.less' : 'btn.more');
    setTimeout(drawWires, 360);
  });
  return [box, btn];
}

function showWhen(el, path, test) {
  bind(path, v => { el.hidden = !test(v); });
  return el;
}

function port(dir, id, labelKey, color) {
  const dot = h('i', { 'data-port': id });
  return h('div', { class: `port ${dir}`, style: `--c: var(--${color})` },
    dir === 'out' ? [t(labelKey), dot] : [dot, t(labelKey)]);
}

function tplBlock(name, tool) {
  const thumb = h('div', { class: 'thumb' });
  const label = h('div', { class: 'name' }, t('opt.template'));
  const el = h('div', { class: 'tpl' }, thumb,
    h('div', { class: 'meta' },
      label,
      h('div', { class: 'btns' },
        h('button', { class: 'btn sm', type: 'button', onclick: () => openCalibrator(tool) }, ic('camera'), t('btn.capture')),
        h('button', { class: 'btn sm icon danger', type: 'button', title: t('btn.clear'), onclick: () => clearTemplate(name) }, ic('trash')))));
  bind('@tpl', () => {
    const src = S.templates[name];
    thumb.style.backgroundImage = src ? `url(${src})` : '';
    thumb.classList.toggle('set', !!src);
    label.textContent = t('opt.template');
    if (src) {
      const img = new Image();
      img.onload = () => { label.append(h('span', { class: 'unit' }, ` · ${img.naturalWidth}×${img.naturalHeight}`)); };
      img.src = src;
    }
  });
  return el;
}

async function clearTemplate(name) {
  const r = await act('clear_template', name);
  if (r.ok) { S.templates = r.templates; notify('@tpl'); }
}

// ── nodes ───────────────────────────────────────────────────────────────────

const NODES = {
  cast: { x: 72, y: 236, w: 262, c: 'yellow', title: 'node.cast' },
  bobber: { x: 388, y: 150, w: 284, c: 'green', title: 'node.bite', test: 'bobber' },
  trigger: { x: 388, y: 486, w: 284, c: 'pink', bare: true },
  reel: { x: 722, y: 214, w: 300, c: 'blue', title: 'node.reel', test: 'reel' },
  preview: { x: 1080, y: 150, w: 300, c: 'blue', title: 'node.preview' },
};

const WIRES = [
  ['cast.cast', 'reel.cast'], ['cast.bobber', 'bobber.in'], ['cast.bite', 'trigger.in'],
  ['bobber.out', 'reel.bobber'], ['trigger.out', 'reel.bite'], ['reel.result', 'preview.in'],
];

const STAGE = {
  countdown: { node: 'cast', wires: [] }, action: { node: 'cast', wires: [] },
  break: { node: 'cast', wires: [] }, cooldown: { node: 'cast', wires: [] },
  cast: { node: 'cast', wires: [1, 2] }, land: { node: 'bobber', wires: [1] },
  bite: { node: 'trigger', wires: [2, 3] }, hook: { node: 'trigger', wires: [4] },
  reel: { node: 'reel', wires: [0, 4, 5] },
};
const NODE_COLOR = { cast: 'yellow', bobber: 'green', trigger: 'pink', reel: 'blue', preview: 'blue' };

function nodeShell(id, body) {
  const def = NODES[id];
  const pill = def.test && h('button', { class: `pill${def.test === 'reel' ? ' blue' : ''}`, type: 'button', onclick: () => probe(def.test) },
    ic('spark'), t('btn.test'));
  const head = def.title && h('div', { class: 'node-head' }, h('span', { class: 'dot' }), h('span', { class: 'title' }, t(def.title)), pill);
  return h('section', { class: `node${def.bare ? ' bare' : ''}`, 'data-node': id, style: `width:${def.w}px; --c: var(--${def.c})` },
    head, body);
}

function buildCast() {
  const chips = h('div', { class: 'chips' });
  bind('cast.points', pts => {
    chips.replaceChildren(...(pts.length ? pts.map((p, i) => h('span', { class: 'chip' }, h('b', {}, `#${i + 1}`), `${p[0]}, ${p[1]}`,
      h('button', { type: 'button', title: t('btn.clear'), onclick: () => setValue('cast.points', pts.filter((_, j) => j !== i)) }, ic('x'))))
      : [h('span', { class: 'note' }, t('points.empty'))]));
  });
  const points = showWhen(h('div', {},
    row('f.cast.points', h('button', { class: 'btn sm', type: 'button', onclick: () => openCalibrator('point') }, ic('pin'), t('btn.pick'))),
    chips), 'cast.target', v => v !== 'cursor');
  const card = h('div', { class: 'card' },
    h('div', { class: 'ports' }, port('out', 'cast.cast', 'port.cast', 'yellow'), port('out', 'cast.bobber', 'port.bobber', 'green'),
      port('out', 'cast.bite', 'port.bite', 'red')),
    row('f.cast.target', select('cast.target', [['cursor', 'opt.cursor'], ['points', 'opt.points'], ['auto', 'opt.auto']])),
    points,
    row('f.cast.power_ms', slider('cast.power_ms'), 'wide'),
    ...fold(
      row('f.cast.power_jitter_ms', stepper('cast.power_jitter_ms')),
      row('f.cast.aim_jitter_px', stepper('cast.aim_jitter_px')),
      row('f.cast.rotate_after', stepper('cast.rotate_after')),
      showWhen(row('f.cast.spot_threshold', stepper('cast.spot_threshold')), 'cast.target', v => v === 'auto')));
  return nodeShell('cast', card);
}

function sessionPanel() {
  const list = h('div', {});
  const update = (i, key, value) => {
    const acts = getPath('session.actions').map(a => ({ ...a }));
    acts[i][key] = value;
    setValue('session.actions', acts);
  };
  bind('session.actions', acts => {
    list.replaceChildren(...(acts.length ? acts.map((a, i) => {
      const key = h('button', { class: 'ctl', type: 'button' }, a.key.toUpperCase());
      key.addEventListener('click', () => {
        key.textContent = '…';
        const onKey = e => {
          e.preventDefault();
          e.stopPropagation();
          window.removeEventListener('keydown', onKey, true);
          const name = e.key === 'Escape' ? null : keyName(e);
          if (name) update(i, 'key', name); else key.textContent = a.key.toUpperCase();
        };
        window.addEventListener('keydown', onKey, true);
      });
      const label = h('input', { class: 'ctl', placeholder: t('action.label'), value: a.label });
      label.addEventListener('change', () => update(i, 'label', label.value.trim()));
      const every = h('input', { class: 'ctl', value: `${a.every_min}`, title: `${t('action.every')} (${UNITS[lang].min})`, inputmode: 'decimal' });
      every.addEventListener('change', () => update(i, 'every_min', clamp(parseFloat(every.value.replace(',', '.')) || 30, 0.5, 600)));
      const on = h('button', { class: `toggle${a.enabled ? ' on' : ''}`, type: 'button', onclick: () => update(i, 'enabled', !a.enabled) });
      const del = h('button', { class: 'x', type: 'button', onclick: () => setValue('session.actions', acts.filter((_, j) => j !== i)) }, ic('x'));
      return h('div', { class: 'action' }, key, label, every, on, del);
    }) : [h('div', { class: 'note' }, t('actions.empty'))]));
  });
  const add = h('button', { class: 'btn sm', type: 'button', onclick: () => {
    const acts = getPath('session.actions');
    if (acts.length < 8) setValue('session.actions', [...acts, { key: String(acts.length + 1), every_min: 30, enabled: true, label: '' }]);
  } }, ic('plus'), t('btn.add'));
  return [
    h('div', { class: 'settings-grid' },
      row('f.session.cooldown_ms', stepper('session.cooldown_ms')),
      row('f.session.cooldown_jitter_ms', stepper('session.cooldown_jitter_ms')),
      row('f.session.max_catches', stepper('session.max_catches')),
      row('f.session.max_minutes', stepper('session.max_minutes')),
      row('f.session.break_every_min', stepper('session.break_every_min')),
      row('f.session.break_minutes', stepper('session.break_minutes')),
      row('f.session.max_fail_streak', stepper('session.max_fail_streak'))),
    h('div', { class: 'sep' }),
    h('h3', {}, t('sub.actions'), h('span', { class: 'grow' }), add),
    list,
  ];
}

function colorRows(sec) {
  return [
    showWhen(row('f.hsv_lo', hsvInput(`${sec}.hsv_lo`)), `${sec}.method`, v => v === 'color'),
    showWhen(row('f.hsv_hi', hsvInput(`${sec}.hsv_hi`)), `${sec}.method`, v => v === 'color'),
    showWhen(row('f.min_area', stepper(`${sec}.min_area`)), `${sec}.method`, v => v === 'color'),
    showWhen(h('div', { class: 'row check' }, h('label', {}, t('f.grayscale')), toggle(`${sec}.grayscale`)), `${sec}.method`, v => v === 'template'),
  ];
}

function buildBobber() {
  const card = h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('span', { class: 'dot', 'data-port': 'bobber.in', style: '--c: var(--green)' }),
      h('span', { class: 'title' }, t('node.bobber')), h('span', { class: 'dot', 'data-port': 'bobber.out', style: '--c: var(--green)' })),
    showWhen(tplBlock('bobber', 'bobber'), 'bite.method', v => v === 'template'),
    row('f.method', select('bite.method', [['template', 'opt.template'], ['color', 'opt.color']])),
    row('f.threshold', slider('bite.threshold'), 'wide'),
    ...fold(...colorRows('bite')));
  return nodeShell('bobber', card);
}

function buildTrigger() {
  const card = h('div', { class: 'card', style: '--c: var(--red)' },
    h('div', { class: 'card-head' }, h('span', { class: 'dot', 'data-port': 'trigger.in' }),
      h('span', { class: 'title' }, t('node.trigger')), h('span', { class: 'dot', 'data-port': 'trigger.out' })),
    row('f.bite.dip_px', stepper('bite.dip_px')),
    row('f.bite.bite_timeout_s', stepper('bite.bite_timeout_s')),
    row('f.bite.hook_delay_ms', stepper('bite.hook_delay_ms')),
    ...fold(
      row('f.bite.confirm_frames', stepper('bite.confirm_frames')),
      row('f.bite.settle_ms', stepper('bite.settle_ms')),
      row('f.bite.appear_timeout_s', stepper('bite.appear_timeout_s')),
      row('f.bite.hook_jitter_ms', stepper('bite.hook_jitter_ms'))));
  return nodeShell('trigger', card);
}

let reelBar = null;

function buildReel() {
  const band = h('div', { class: 'band' });
  const edgeL = h('div', { class: 'edge l' });
  const edgeR = h('div', { class: 'edge r' });
  const marker = h('div', { class: 'marker' });
  reelBar = h('div', { class: 'reelbar' }, edgeL, edgeR, band, marker, h('div', { class: 'label' }, h('span', {}, t('reel.waiting'))));
  reelBar.marker = marker;
  const paint = () => {
    const r = S.cfg.reel;
    band.style.left = `${(r.target - r.deadband / 2) * 100}%`;
    band.style.width = `${Math.max(0.4, r.deadband * 100)}%`;
    edgeL.style.width = edgeR.style.width = `${r.edge_guard * 100}%`;
  };
  ['reel.target', 'reel.deadband', 'reel.edge_guard'].forEach(p => bind(p, paint));
  const card = h('div', { class: 'card' },
    h('div', { class: 'ports io' },
      h('div', { class: 'ports', style: 'margin:0' }, port('in', 'reel.cast', 'port.cast', 'yellow'),
        port('in', 'reel.bobber', 'port.bobber', 'green'), port('in', 'reel.bite', 'port.bite', 'red')),
      port('out', 'reel.result', 'port.result', 'blue')),
    reelBar,
    showWhen(tplBlock('marker', 'marker'), 'reel.method', v => v === 'template'),
    row('f.reel.control', select('reel.control', [['predictive', 'opt.predictive'], ['hysteresis', 'opt.hysteresis']])),
    row('f.reel.target', stepper('reel.target')),
    row('f.reel.deadband', stepper('reel.deadband')),
    showWhen(row('f.reel.lookahead_ms', stepper('reel.lookahead_ms')), 'reel.control', v => v === 'predictive'),
    row('f.reel.hold_moves', select('reel.hold_moves', [['right', 'opt.right'], ['left', 'opt.left']])),
    row('f.reel.fps', stepper('reel.fps')),
    ...fold(
      row('f.reel.edge_guard', stepper('reel.edge_guard')),
      row('f.reel.min_toggle_ms', stepper('reel.min_toggle_ms')),
      row('f.method', select('reel.method', [['template', 'opt.template'], ['color', 'opt.color']])),
      row('f.threshold', slider('reel.threshold'), 'wide'),
      row('f.reel.appear_timeout_s', stepper('reel.appear_timeout_s')),
      row('f.reel.end_confirm_ms', stepper('reel.end_confirm_ms')),
      row('f.reel.max_duration_s', stepper('reel.max_duration_s')),
      ...colorRows('reel')));
  return nodeShell('reel', card);
}

let previewEls = null;

function buildPreview() {
  const img = h('img', { alt: '' });
  const backdrop = h('img', { alt: '', class: 'backdrop' });
  const badge = h('div', { class: 'badge' }, h('span', { class: 'live' }), h('span', {}, t('preview.live')));
  const title = h('h4', {}, t('preview.idle'));
  const text = h('p', {}, t('preview.hint'));
  const frame = h('div', { class: 'preview-frame' }, h('div', { class: 'aurora' }), backdrop, img, badge, h('div', { class: 'caption' }, title, text));
  frame.insertAdjacentHTML('beforeend', GLYPH);
  frame.append(frame.querySelector('.caption'));
  img.hidden = backdrop.hidden = true;
  badge.hidden = true;
  const tb = (iconName, tip, fn) => h('button', { class: 'tb', type: 'button', title: t(tip), onclick: fn }, ic(iconName));
  const live = tb('eye', 'preview.live', () => { S.preview.live = !S.preview.live; live.classList.toggle('on', S.preview.live); });
  live.classList.toggle('on', S.preview.live);
  const toolbar = h('div', { class: 'toolbar' },
    tb('expand', 'tip.expand', () => openViewer()),
    tb('target', 'tip.probe_bobber', () => probe('bobber')),
    tb('bar', 'tip.probe_reel', () => probe('reel')),
    tb('bubbles', 'tip.probe_spot', () => probe('spot')),
    h('span', { class: 'grow' }), live);
  const card = h('div', { class: 'card', style: 'padding:10px' },
    h('div', { class: 'ports', style: 'margin: 0 2px 10px' }, port('in', 'preview.in', 'port.vision', 'blue')),
    frame);
  previewEls = { img, backdrop, badge, title, text, frame };
  return nodeShell('preview', h('div', {}, card, toolbar));
}

function renderNodes() {
  binders.clear();
  const root = $('#nodes');
  root.replaceChildren(buildCast(), buildBobber(), buildTrigger(), buildReel(), buildPreview());
  for (const el of $$('.node', root)) {
    placeNode(el);
    new ResizeObserver(() => drawWires()).observe(el);
    const handle = $('.node-head', el) || $('.card-head', el);
    if (handle) dragNode(el, handle);
  }
  drawWires();
  updateStage();
}

function nodePos(id) {
  const saved = S.layout[id];
  return saved ? { x: saved[0], y: saved[1] } : { x: NODES[id].x, y: NODES[id].y };
}

function placeNode(el) {
  const p = nodePos(el.dataset.node);
  el.style.left = `${p.x}px`;
  el.style.top = `${p.y}px`;
}

function dragNode(el, handle) {
  handle.addEventListener('pointerdown', e => {
    if (e.button !== 0 || e.target.closest('button, input, select')) return;
    e.preventDefault();
    const id = el.dataset.node;
    const start = nodePos(id);
    const [sx, sy] = [e.clientX, e.clientY];
    el.classList.add('dragging');
    handle.setPointerCapture(e.pointerId);
    const move = ev => {
      S.layout[id] = [Math.round(start.x + (ev.clientX - sx) / S.zoom), Math.round(start.y + (ev.clientY - sy) / S.zoom)];
      placeNode(el);
      drawWires();
      updateCursor();
    };
    const up = () => {
      el.classList.remove('dragging');
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      saveUi();
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
  });
}

// ── wires & canvas ──────────────────────────────────────────────────────────

let wireFrame = 0;

function portCenter(id) {
  const el = $(`[data-port="${id}"]`);
  if (!el || !el.offsetParent) return null;
  const r = el.getBoundingClientRect();
  const w = $('#world').getBoundingClientRect();
  return [(r.left + r.width / 2 - w.left) / S.zoom, (r.top + r.height / 2 - w.top) / S.zoom];
}

function drawWires() {
  cancelAnimationFrame(wireFrame);
  wireFrame = requestAnimationFrame(() => {
    const g = $('#wire-paths');
    const active = new Set(STAGE[S.snap?.stage]?.wires || []);
    g.replaceChildren(...WIRES.map(([a, b], i) => {
      const p = portCenter(a), q = portCenter(b);
      if (!p || !q) return document.createComment('');
      const dx = Math.max(40, Math.abs(q[0] - p[0]) * 0.5);
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M${p[0]},${p[1]} C${p[0] + dx},${p[1]} ${q[0] - dx},${q[1]} ${q[0]},${q[1]}`);
      if (active.has(i) && S.snap?.status === 'running') path.classList.add('flow');
      return path;
    }));
  });
}

function applyView() {
  const [x, y] = S.pan;
  $('#world').style.transform = `translate(${x}px, ${y}px) scale(${S.zoom})`;
  const c = $('#canvas');
  c.style.backgroundSize = `${24 * S.zoom}px ${24 * S.zoom}px`;
  c.style.backgroundPosition = `${x}px ${y}px`;
}

function zoomAt(factor, cx = innerWidth / 2, cy = innerHeight / 2) {
  const z = clamp(S.zoom * factor, 0.4, 1.6);
  const k = z / S.zoom;
  S.pan = [cx - (cx - S.pan[0]) * k, cy - (cy - S.pan[1]) * k];
  S.zoom = z;
  applyView();
  saveUi();
}

function fit() {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const el of $$('.node')) {
    const p = nodePos(el.dataset.node);
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y - 36);
    x1 = Math.max(x1, p.x + el.offsetWidth); y1 = Math.max(y1, p.y + el.offsetHeight);
  }
  const top = 118, bottom = 150, side = 76;
  const z = clamp(Math.min((innerWidth - side * 2) / (x1 - x0), (innerHeight - top - bottom) / (y1 - y0)), 0.4, 1);
  S.zoom = z;
  S.pan = [(innerWidth - (x1 - x0) * z) / 2 - x0 * z, top + (innerHeight - top - bottom - (y1 - y0) * z) / 2 - y0 * z];
  applyView();
  saveUi();
}

function initCanvas() {
  const canvas = $('#canvas');
  canvas.addEventListener('pointerdown', e => {
    if (e.target.closest('.node')) return;
    const [sx, sy] = [e.clientX, e.clientY];
    const [px, py] = S.pan;
    canvas.classList.add('panning');
    canvas.setPointerCapture(e.pointerId);
    const move = ev => { S.pan = [px + ev.clientX - sx, py + ev.clientY - sy]; applyView(); };
    const up = () => {
      canvas.classList.remove('panning');
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      saveUi();
    };
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
  });
  canvas.addEventListener('wheel', e => {
    if (e.target.closest('select, .panel')) return;
    e.preventDefault();
    zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY);
  }, { passive: false });
  $$('[data-zoom]').forEach(b => b.addEventListener('click', () => {
    const z = b.dataset.zoom;
    if (z === 'in') zoomAt(1.15);
    else if (z === 'out') zoomAt(1 / 1.15);
    else if (z === 'fit') fit();
    else if (z === 'wires') {
      S.wires = !S.wires;
      document.body.classList.toggle('no-wires', !S.wires);
      b.classList.toggle('on', S.wires);
      saveUi();
    } else if (z === 'layout') resetLayout();
  }));
  addEventListener('resize', () => drawWires());
}

function resetLayout() {
  S.layout = {};
  $$('.node').forEach(placeNode);
  fit();
  drawWires();
}

let uiTimer = 0;

function saveUi() {
  clearTimeout(uiTimer);
  uiTimer = setTimeout(() => call('save_ui', {
    lang, layout: S.layout, zoom: +S.zoom.toFixed(3), pan: S.pan.map(Math.round), wires: S.wires, dock: S.dock,
  }), 600);
}

// ── live state ──────────────────────────────────────────────────────────────

function fmtClock(s) {
  s = Math.floor(s || 0);
  return [s / 3600, (s % 3600) / 60, s % 60].map(v => String(Math.floor(v)).padStart(2, '0')).join(':');
}

function eventText(e) {
  const p = { ...e.params };
  if (e.code === 'vision_error') return t(`err.${p.code}`, { name: p.name ? t(`name.${p.name}`) : '' });
  if (p.reason) p.reason = t(`reason.${p.reason}`);
  return t(`ev.${e.code}`, p);
}

function stageLabel(snap) {
  if (snap.status === 'paused') return snap.pause_reason === 'focus' ? t('paused.focus') : t('stage.paused');
  return t(`stage.${snap.stage}`);
}

function applySnap(snap) {
  const prevStage = S.snap?.stage, prevStatus = S.snap?.status;
  S.snap = snap;
  if (snap.logs.length) {
    S.logs.push(...snap.logs);
    if (S.logs.length > 400) S.logs.splice(0, S.logs.length - 400);
    S.lastLog = snap.logs.at(-1).id;
    renderLogCard();
    if (S.panel === 'log') appendLog(snap.logs);
    for (const e of snap.logs) if (e.level === 'error') toast(eventText(e), 'error');
  }
  document.body.classList.toggle('running', snap.status === 'running');
  document.body.classList.toggle('paused', snap.status === 'paused');
  const run = $('#run');
  if (run.dataset.state !== snap.status + lang) {
    run.dataset.state = snap.status + lang;
    run.classList.toggle('running', snap.status === 'running');
    run.querySelector('svg').replaceWith(ic(snap.status === 'running' ? 'pause' : 'play'));
    run.querySelector('.lbl').textContent = t(snap.status === 'running' ? 'btn.pause' : snap.status === 'paused' ? 'btn.resume' : 'btn.start');
  }
  $('#stage').textContent = stageLabel(snap);
  $('#remain').textContent = snap.remaining ? `· ${Math.ceil(snap.remaining)}${UNITS[lang].s}` : '';
  $('#count').textContent = snap.stats.catches;
  renderCorners(snap.stats);
  if (reelBar) {
    const live = snap.stage === 'reel' && snap.reel_x != null;
    reelBar.classList.toggle('live', live);
    reelBar.classList.toggle('hold', live && snap.hold);
    if (live) reelBar.marker.style.left = `${snap.reel_x * 100}%`;
  }
  if (prevStage !== snap.stage || prevStatus !== snap.status) updateStage();
  if (S.panel === 'stats') renderPanel();
}

function updateStage() {
  const snap = S.snap;
  const info = snap && snap.status === 'running' ? STAGE[snap.stage] : null;
  $$('.node').forEach(el => el.classList.toggle('active', !!info && el.dataset.node === info.node));
  updateCursor();
  drawWires();
}

function updateCursor() {
  const bot = $('#bot');
  const snap = S.snap;
  const info = snap && snap.status === 'running' ? STAGE[snap.stage] : null;
  const el = info && $(`.node[data-node="${info.node}"]`);
  if (!el) { bot.classList.add('off'); return; }
  const p = nodePos(info.node);
  bot.classList.remove('off');
  bot.style.setProperty('--c', `var(--${NODE_COLOR[info.node]})`);
  bot.style.transform = `translate(${p.x + el.offsetWidth * 0.6}px, ${p.y + el.offsetHeight - 26}px)`;
  bot.querySelector('.tag').textContent = stageLabel(snap);
}

function renderCorners(s) {
  const line = (k, v) => h('div', { title: t(`corner.${k}`) }, `${k}: `, h('b', {}, v));
  $('#corner-l').replaceChildren(line('T', fmtClock(s.active_s)), line('C', s.catches), line('H', `${s.per_hour}`),
    line('S', `${s.success}%`));
  $('#corner-r').replaceChildren(line('F', s.fps), line('B', `${s.avg_bite}${UNITS[lang].s}`),
    line('R', `${s.avg_reel}${UNITS[lang].s}`), line('M', s.misses));
}

function renderLogCard() {
  const [prev, last] = S.logs.slice(-2);
  $('#log-last').textContent = last ? eventText(last) : t('log.empty');
  $('#log-prev').textContent = prev ? eventText(prev) : '';
}

let polling = false;

async function poll() {
  if (polling) return;
  polling = true;
  const r = await call('state', S.lastLog);
  polling = false;
  if (r.ok) applySnap(r);
  const busy = r.ok && r.status !== 'idle';
  setTimeout(poll, !r.ok ? 1500 : r.stage === 'reel' ? 60 : busy ? 200 : 700);
}

async function previewLoop() {
  const snap = S.snap;
  if (S.preview.live && snap && snap.status === 'running' && !document.hidden) {
    const r = await call('preview');
    if (r.ok && r.image) showFrame(r.image, true, t(`stage.${snap.stage}`), S.logs.length ? eventText(S.logs.at(-1)) : '');
  }
  setTimeout(previewLoop, 110);
}

function showFrame(src, live, title, text) {
  if (!previewEls) return;
  const { img, backdrop, badge, frame } = previewEls;
  img.src = backdrop.src = src;
  img.hidden = backdrop.hidden = false;
  badge.hidden = !live;
  frame.classList.add('has-image');
  previewEls.title.textContent = title;
  previewEls.text.textContent = text;
  S.preview.src = src;
}

async function probe(kind) {
  const r = await act('probe', kind);
  if (!r.ok) return;
  const status = r.found ? t('preview.found') : t('preview.notfound');
  const detail = kind === 'spot' ? t('preview.spots', { n: r.count }) : t('preview.score', { s: r.score.toFixed(2) });
  showFrame(r.image, false, `${t('preview.test')} · ${t(`name.${kind === 'reel' ? 'marker' : kind}`)}`, `${status} · ${detail}`);
  toast(`${t(`name.${kind === 'reel' ? 'marker' : kind}`)}: ${status} · ${detail}`, r.found ? 'ok' : 'warn');
}

// ── dock & panels ───────────────────────────────────────────────────────────

function togglePanel(name) {
  S.panel = S.panel === name ? null : name;
  $$('.dock-tabs [data-panel]').forEach(b => b.classList.toggle('on', b.dataset.panel === S.panel));
  $('#panel').classList.toggle('open', !!S.panel);
  if (S.panel) renderPanel();
}

function appendLog(events) {
  const list = $('#panel .loglist');
  if (!list) return;
  const atBottom = $('#panel').scrollHeight - $('#panel').scrollTop - $('#panel').clientHeight < 40;
  list.append(...events.map(logRow));
  if (atBottom) $('#panel').scrollTop = $('#panel').scrollHeight;
}

function logRow(e) {
  const time = new Date(e.ts * 1000).toLocaleTimeString(lang, { hour12: false });
  return h('div', { class: `e ${e.level}` }, h('time', {}, time), h('i'), h('span', {}, eventText(e)));
}

function renderPanel() {
  const panel = $('#panel');
  unbind(S.panelBinds);
  scope = [];
  const title = (key, ...extra) => h('h3', {}, t(key), h('span', { class: 'grow' }), ...extra);
  const kids = [];
  if (S.panel === 'log') {
    kids.push(title('tab.log'), h('div', { class: 'loglist' }, S.logs.length ? S.logs.map(logRow) : h('div', { class: 'note' }, t('log.empty'))));
  } else if (S.panel === 'stats') {
    const s = S.snap?.stats || {};
    const cards = [['catches', s.catches], ['casts', s.casts], ['success', `${s.success ?? 0}%`], ['per_hour', s.per_hour],
      ['escaped', s.escaped], ['misses', s.misses], ['avg_bite', `${s.avg_bite ?? 0}${UNITS[lang].s}`],
      ['avg_reel', `${s.avg_reel ?? 0}${UNITS[lang].s}`], ['best_streak', s.best_streak], ['fail_streak', s.fail_streak],
      ['fps', s.fps], ['active', fmtClock(s.active_s)]];
    const reels = S.logs.filter(e => e.code === 'caught').slice(-40).map(e => +e.params.s);
    kids.push(title('tab.stats', h('button', { class: 'btn sm', onclick: () => act('reset_stats') }, ic('refresh'), t('tip.reset_stats'))),
      h('div', { class: 'grid-stats' }, cards.map(([k, v]) => h('div', { class: 'stat' }, h('div', { class: 'v' }, v ?? 0), h('div', { class: 'k' }, t(`stat.${k}`))))),
      spark(reels));
  } else if (S.panel === 'templates') {
    const tools = { bobber: 'bobber', marker: 'marker', spot: 'spot' };
    kids.push(title('tab.templates'), h('div', { class: 'tpl-grid' }, Object.entries(tools).map(([name, tool]) => {
      const src = S.templates[name];
      return h('div', { class: 'tpl-card' },
        h('div', { class: `thumb${src ? ' set' : ''}`, style: src ? `background-image:url(${src})` : '' }),
        h('div', { class: 'n' }, h('span', { class: 'dot', style: `--c: var(--${{ bobber: 'green', marker: 'blue', spot: 'yellow' }[name]})` }), t(`tool.${tool}`)),
        h('div', { style: 'display:flex; gap:6px' },
          h('button', { class: 'btn sm', onclick: () => openCalibrator(tool) }, ic('camera'), t('btn.capture')),
          src && h('button', { class: 'btn sm danger', onclick: async () => { await clearTemplate(name); renderPanel(); } }, ic('trash'))));
    })));
  } else if (S.panel === 'profiles') {
    kids.push(title('tab.profiles'), h('div', { class: 'plist' }, S.boot.profiles.map(name =>
      h('button', { class: `p${name === S.boot.profile ? ' cur' : ''}`, onclick: () => switchProfile(name) },
        ic('folder'), h('span', { class: 'grow' }, name), name === S.boot.profile && ic('check')))),
      h('div', { class: 'actions' },
        h('button', { class: 'btn sm', onclick: () => profileOp('new') }, ic('plus'), t('wf.new')),
        h('button', { class: 'btn sm', onclick: () => profileOp('duplicate') }, ic('copy'), t('wf.duplicate')),
        h('button', { class: 'btn sm', onclick: () => profileOp('rename') }, ic('edit'), t('wf.rename')),
        h('button', { class: 'btn sm danger', onclick: () => profileOp('delete') }, ic('trash'), t('wf.delete'))));
  } else if (S.panel === 'session') {
    kids.push(title('node.session', h('button', { class: 'btn sm', onclick: () => resetSection('session') }, ic('refresh'), t('btn.reset'))),
      ...sessionPanel());
  } else if (S.panel === 'settings') {
    const monitors = Array.from({ length: Math.max(1, S.boot.monitors) }, (_, i) => [String(i + 1), String(i + 1)]);
    const mon = select('system.monitor', monitors, Number);
    kids.push(title('tab.settings', h('button', { class: 'btn sm', onclick: () => act('open_folder') }, ic('folder'), t('wf.folder'))),
      h('div', { class: 'settings-grid' },
        row('f.system.hotkey_toggle', keyInput('system.hotkey_toggle')),
        row('f.system.hotkey_stop', keyInput('system.hotkey_stop')),
        row('f.system.start_delay_s', stepper('system.start_delay_s')),
        row('f.system.idle_fps', stepper('system.idle_fps')),
        row('f.system.window_title', textInput('system.window_title')),
        row('f.system.monitor', mon),
        h('div', { class: 'row check' }, h('label', {}, t('f.system.require_focus')), toggle('system.require_focus')),
        h('div', { class: 'row check' }, h('label', {}, t('f.system.failsafe')), toggle('system.failsafe')),
        h('div', { class: 'row check' }, h('label', {}, t('f.system.sound')), toggle('system.sound'))),
      h('div', { class: 'sep' }),
      h('div', { class: 'note' }, S.boot.data_dir));
  }
  S.panelBinds = scope;
  scope = null;
  panel.replaceChildren(...kids);
  if (S.panel === 'log') panel.scrollTop = panel.scrollHeight;
}

async function resetSection(section) {
  await flush();
  const r = await act('reset_section', section);
  if (r.ok) applyConfig(r.config);
}

function spark(values) {
  if (values.length < 2) return null;
  const box = h('div', { class: 'spark' });
  const max = Math.max(...values) * 1.1, min = 0;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${38 - ((v - min) / (max - min)) * 36}`);
  box.innerHTML = `<svg viewBox="0 0 100 38" preserveAspectRatio="none">
    <defs><linearGradient id="sg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#5aa8ff" stop-opacity=".35"/><stop offset="1" stop-color="#5aa8ff" stop-opacity="0"/></linearGradient></defs>
    <path d="M0,38 L${pts.join(' L')} L100,38 Z" fill="url(#sg)"/>
    <polyline points="${pts.join(' ')}" fill="none" stroke="#5aa8ff" stroke-width="1.4" vector-effect="non-scaling-stroke"/></svg>`;
  return box;
}

// ── menus, toasts, dialogs ──────────────────────────────────────────────────

function openMenu(anchor, items) {
  closeMenu();
  const menu = h('div', { class: 'menu' }, items.map(it => it === '-' ? h('div', { class: 'hr' })
    : h('button', { class: it.danger ? 'danger' : '', onclick: () => { closeMenu(); it.fn(); } }, ic(it.icon), t(it.label),
      it.hint && h('span', { class: 'hint' }, it.hint))));
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  const left = Math.min(r.left, innerWidth - menu.offsetWidth - 12);
  menu.style.left = `${left}px`;
  menu.style.top = `${r.bottom + 8}px`;
  setTimeout(() => addEventListener('pointerdown', outside, true), 0);
  function outside(e) { if (!menu.contains(e.target)) closeMenu(); }
  menu.cleanup = () => removeEventListener('pointerdown', outside, true);
}

function closeMenu() {
  $$('.menu').forEach(m => { m.cleanup?.(); m.remove(); });
}

const MENUS = {
  workflow: () => [
    { icon: 'plus', label: 'wf.new', fn: () => profileOp('new') },
    { icon: 'copy', label: 'wf.duplicate', fn: () => profileOp('duplicate') },
    { icon: 'edit', label: 'wf.rename', fn: () => profileOp('rename') },
    '-',
    { icon: 'folder', label: 'wf.folder', fn: () => act('open_folder') },
    { icon: 'trash', label: 'wf.delete', fn: () => profileOp('delete'), danger: true },
  ],
  help: () => [
    { icon: 'spark', label: 'help.guide', fn: showGuide },
    { icon: 'gear', label: 'help.hotkeys', fn: showHotkeys },
    { icon: 'checkCircle', label: 'help.about', fn: showAbout },
  ],
  more: () => [
    { icon: 'checkCircle', label: 'btn.check', fn: showCheck },
    { icon: 'refresh', label: 'wf.reset_stats', fn: () => act('reset_stats') },
    { icon: 'layout', label: 'wf.layout', fn: resetLayout },
    { icon: 'folder', label: 'wf.folder', fn: () => act('open_folder') },
  ],
};

function toast(text, kind = 'ok') {
  const el = h('div', { class: `toast ${kind}` }, h('i'), h('span', {}, text));
  $('#toasts').append(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 320); }, kind === 'error' ? 4200 : 2400);
}

function dialog(titleKey, body, buttons = [['btn.ok', true]], wide = false) {
  return new Promise(resolve => {
    const close = v => { modal.remove(); removeEventListener('keydown', onKey, true); resolve(v); };
    const onKey = e => { if (e.key === 'Escape') { e.stopPropagation(); close(null); } };
    const modal = h('div', { class: 'modal dialog', onpointerdown: e => { if (e.target === modal) close(null); } },
      h('div', { class: `box${wide ? ' wide' : ''}` }, h('h3', {}, t(titleKey)), body,
        h('div', { class: 'actions' }, buttons.map(([k, v, primary]) =>
          h('button', { class: `btn sm${primary ? ' primary' : ''}`, onclick: () => close(typeof v === 'function' ? v() : v) }, t(k))))));
    document.body.append(modal);
    addEventListener('keydown', onKey, true);
    $('input', modal)?.focus();
  });
}

async function ask(titleKey, value = '') {
  const input = h('input', { class: 'ctl', value });
  input.addEventListener('keydown', e => { if (e.key === 'Enter') input.closest('.box').querySelector('.primary').click(); });
  const v = await dialog(titleKey, input, [['btn.cancel', null], ['btn.ok', () => input.value.trim(), true]]);
  return v || null;
}

function showGuide() {
  const keys = { toggle: S.cfg.system.hotkey_toggle.toUpperCase(), stop: S.cfg.system.hotkey_stop.toUpperCase() };
  dialog('guide.title', h('div', { class: 'guide' }, h('ol', {}, [1, 2, 3, 4, 5].map(i => h('li', {}, t(`guide.${i}`, keys))))),
    [['btn.calibrate', () => openCalibrator()], ['btn.ok', true, true]], true);
}

function showHotkeys() {
  const s = S.cfg.system;
  dialog('help.hotkeys', h('div', { class: 'hk' },
    h('kbd', {}, s.hotkey_toggle.toUpperCase()), t('hotkeys.toggle'),
    h('kbd', {}, s.hotkey_stop.toUpperCase()), t('hotkeys.stop'),
    h('kbd', {}, '↖'), t('hotkeys.failsafe')));
}

function showAbout() {
  dialog('help.about', h('p', {}, `Albion Fishing Bot v${S.boot.version}. `, t('about.text')));
}

async function showCheck() {
  const c = S.cfg, tp = S.templates, r = c.regions;
  const ok = x => x.width >= 4 && x.height >= 4;
  const target = c.cast.target === 'cursor' ? [true, t('check.cursor')]
    : c.cast.target === 'points' ? [c.cast.points.length > 0, t('check.points', { n: c.cast.points.length })]
    : [ok(r.water) && !!tp.spot, t('check.auto')];
  const items = [
    ['check.bobber_area', ok(r.bobber), r.bobber.width ? `${r.bobber.width}×${r.bobber.height}` : ''],
    ['check.bobber', c.bite.method === 'color' || !!tp.bobber, c.bite.method === 'color' ? t('check.color') : ''],
    ['check.reel', ok(r.reel), r.reel.width ? `${r.reel.width}×${r.reel.height}` : ''],
    ['check.marker', c.reel.method === 'color' || !!tp.marker, c.reel.method === 'color' ? t('check.color') : ''],
    ['check.target', ...target],
    ['check.hotkeys', S.boot.hotkeys, ''],
  ];
  const server = await call('check');
  const ready = items.every(i => i[1]) && server.ok;
  const body = h('div', {},
    h('div', { class: 'checklist' }, items.map(([k, good, d]) => h('div', { class: `c${good ? ' ok' : ''}` },
      h('span', { class: 's' }, ic(good ? 'check' : 'x')), h('span', { class: 'grow' }, t(k)), h('span', { class: 'd' }, d)))),
    ready ? h('div', { class: 'ready' }, t('check.ready', { key: c.system.hotkey_toggle.toUpperCase() }))
      : !server.ok && server.error !== 'offline' && h('div', { class: 'note', style: 'margin-top:12px; color: var(--red)' }, errText(server)));
  dialog('check.title', body, ready ? [['btn.ok', true, true]] : [['btn.ok', true], ['btn.calibrate', () => openCalibrator(), true]], true);
}

function openViewer() {
  if (!S.preview.src) return;
  const img = h('img', { src: S.preview.src, style: 'max-width: 92vw; max-height: 86vh; border-radius: 14px; image-rendering: pixelated' });
  const modal = h('div', { class: 'modal dialog', onclick: () => modal.remove() }, img);
  document.body.append(modal);
}

// ── profiles ────────────────────────────────────────────────────────────────

function applyBoot(boot) {
  S.boot = boot;
  S.schema = boot.schema;
  S.cfg = boot.config;
  S.templates = boot.templates;
  $('#profile-name').textContent = boot.profile;
  $('#version').textContent = boot.version;
  $('#demo-tag').hidden = !boot.demo;
  $('#kbd-toggle').textContent = boot.config.system.hotkey_toggle.toUpperCase();
  renderNodes();
  if (S.panel) renderPanel();
}

async function switchProfile(name) {
  if (name === S.boot.profile) return;
  await flush();
  const r = await act('profile_switch', name);
  if (r.ok) { applyBoot(r); toast(t('toast.profile', { name })); }
}

async function profileOp(op) {
  await flush();
  let r;
  if (op === 'new' || op === 'duplicate') {
    const name = await ask('prompt.name', op === 'new' ? 'Profile' : `${S.boot.profile} 2`);
    if (!name) return;
    r = await act('profile_create', name, op === 'duplicate');
  } else if (op === 'rename') {
    const name = await ask('prompt.name', S.boot.profile);
    if (!name) return;
    r = await act('profile_rename', name);
  } else if (op === 'delete') {
    const yes = await dialog('wf.delete', h('p', {}, t('confirm.delete', { name: S.boot.profile })), [['btn.cancel', false], ['wf.delete', true, true]]);
    if (!yes) return;
    r = await act('profile_delete');
  }
  if (r?.ok) { applyBoot(r); toast(t('toast.profile', { name: r.profile })); }
}

function cycleProfile(dir) {
  const list = S.boot.profiles;
  const i = list.indexOf(S.boot.profile);
  switchProfile(list[(i + dir + list.length) % list.length]);
}

// ── calibrator ──────────────────────────────────────────────────────────────

const TOOLS = [
  { id: 'bobber_area', kind: 'region', name: 'bobber', c: 'green' },
  { id: 'bobber', kind: 'template', name: 'bobber', c: 'green' },
  { id: 'reel', kind: 'region', name: 'reel', c: 'blue' },
  { id: 'marker', kind: 'template', name: 'marker', c: 'blue' },
  { id: 'water', kind: 'region', name: 'water', c: 'yellow' },
  { id: 'spot', kind: 'template', name: 'spot', c: 'yellow' },
  { id: 'point', kind: 'point', c: 'red' },
];
const CAL = { shot: null, tool: 'bobber_area', zoom: 1, ox: 0, oy: 0, el: null };

function toolDone(tool) {
  const r = S.cfg.regions;
  if (tool.kind === 'region') return r[tool.name].width >= 4;
  if (tool.kind === 'template') return !!S.templates[tool.name];
  return S.cfg.cast.points.length > 0;
}

function openCalibrator(toolId) {
  if (toolId) CAL.tool = toolId;
  CAL.el?.remove();
  const stage = h('div', { class: 'cal-stage' });
  const layer = h('div', { class: 'layer' });
  const tools = h('div', { class: 'tools' });
  const hint = h('div', { class: 'hint' });
  const coords = h('div', { class: 'coords' });
  const delay = h('select', { class: 'ctl', style: 'width:auto' }, [0, 3, 5].map(s => h('option', { value: s }, `${s}${UNITS[lang].s}`)));
  delay.value = S.boot.native ? '0' : '3';
  const close = () => { CAL.el.remove(); CAL.el = null; removeEventListener('keydown', onKey, true); drawWires(); };
  const onKey = e => { if (e.key === 'Escape' && !$('.dialog')) close(); };
  addEventListener('keydown', onKey, true);
  CAL.el = h('div', { class: 'modal cal' },
    h('div', { class: 'cal-top' }, h('span', { class: 'title' }, t('cal.title')), tools, h('span', { style: 'flex:1' }),
      delay, h('button', { class: 'btn', onclick: () => capture(+delay.value) }, ic('camera'), t('cal.capture')),
      h('button', { class: 'btn primary', onclick: close }, t('cal.done'))),
    stage,
    h('div', { class: 'cal-bottom' }, hint, h('span', {}, t('cal.zoom')), coords));
  stage.append(layer);
  document.body.append(CAL.el);
  CAL.stage = stage; CAL.layer = layer; CAL.tools = tools; CAL.hint = hint; CAL.coords = coords; CAL.delay = delay;

  const renderTools = () => {
    tools.replaceChildren(...TOOLS.map(tl => h('button', { class: tl.id === CAL.tool ? 'on' : '', onclick: () => { CAL.tool = tl.id; renderTools(); renderShapes(); } },
      h('span', { class: 'dot', style: `--c: var(--${tl.c})` }), t(`tool.${tl.id}`), toolDone(tl) && ic('check'))));
    $$('svg', tools).forEach(s => s.classList.add('ok'));
    hint.textContent = t(`hint.${CAL.tool}`);
  };
  CAL.renderTools = renderTools;
  renderTools();
  stageEvents(stage);
  if (CAL.shot) mountShot(); else showEmpty();
}

function showEmpty() {
  CAL.stage.querySelector('.cal-empty')?.remove();
  CAL.stage.append(h('div', { class: 'cal-empty' }, h('div', {},
    h('div', { class: 'big' }, t('cal.empty')),
    h('button', { class: 'btn primary', onclick: () => capture(+CAL.delay.value) }, ic('camera'), t('cal.capture')))));
}

async function capture(delay) {
  CAL.stage.querySelector('.cal-empty')?.remove();
  let cd = null;
  if (delay > 0) {
    cd = h('div', { class: 'countdown' });
    CAL.stage.append(cd);
    for (let s = delay; s > 0; s--) { cd.textContent = t('cal.countdown', { s }); await new Promise(r => setTimeout(r, 1000)); }
  }
  const r = await act('capture', 0);
  cd?.remove();
  if (!r.ok) { if (!CAL.shot) showEmpty(); return; }
  CAL.shot = r;
  mountShot(true);
}

function mountShot(refit) {
  const img = h('img', { src: CAL.shot.image, width: CAL.shot.width, height: CAL.shot.height, draggable: 'false' });
  CAL.layer.replaceChildren(img);
  img.decode?.().catch(() => {}).finally(() => {
    if (refit || !CAL.fitted) calFit();
    renderShapes();
  });
}

function calFit() {
  const r = CAL.stage.getBoundingClientRect();
  CAL.zoom = Math.min(r.width / CAL.shot.width, r.height / CAL.shot.height);
  CAL.ox = (r.width - CAL.shot.width * CAL.zoom) / 2;
  CAL.oy = (r.height - CAL.shot.height * CAL.zoom) / 2;
  CAL.fitted = true;
  calView();
}

function calView() {
  CAL.layer.style.transform = `translate(${CAL.ox}px, ${CAL.oy}px) scale(${CAL.zoom})`;
  CAL.layer.style.setProperty('--inv', 1 / CAL.zoom);
  $$('.shape span', CAL.layer).forEach(s => { s.style.transform = `scale(${1 / CAL.zoom})`; });
  $$('.shape', CAL.layer).forEach(s => { s.style.borderWidth = `${1.5 / CAL.zoom}px`; });
  $$('.pt', CAL.layer).forEach(s => { s.style.transform = `scale(${1 / CAL.zoom})`; });
}

function renderShapes() {
  if (!CAL.shot) return;
  $$('.shape, .pt', CAL.layer).forEach(el => el.remove());
  const { left, top } = CAL.shot;
  for (const tl of TOOLS.filter(x => x.kind === 'region')) {
    const r = S.cfg.regions[tl.name];
    if (r.width < 4) continue;
    CAL.layer.append(h('div', { class: `shape${tl.id === 'water' ? ' below' : ''}`, style: `--c: var(--${tl.c}); left:${r.left - left}px; top:${r.top - top}px; width:${r.width}px; height:${r.height}px; opacity:${tl.id === CAL.tool ? 1 : .55}` },
      h('span', {}, t(`tool.${tl.id}`))));
  }
  S.cfg.cast.points.forEach(([x, y]) => CAL.layer.append(h('div', { class: 'pt', style: `left:${x - left}px; top:${y - top}px` })));
  calView();
}

function stageEvents(stage) {
  const toImg = e => {
    const r = stage.getBoundingClientRect();
    return [(e.clientX - r.left - CAL.ox) / CAL.zoom, (e.clientY - r.top - CAL.oy) / CAL.zoom];
  };
  stage.addEventListener('contextmenu', e => e.preventDefault());
  stage.addEventListener('wheel', e => {
    if (!CAL.shot) return;
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    const z = clamp(CAL.zoom * Math.exp(-e.deltaY * 0.0018), 0.05, 12);
    CAL.ox = cx - (cx - CAL.ox) * (z / CAL.zoom);
    CAL.oy = cy - (cy - CAL.oy) * (z / CAL.zoom);
    CAL.zoom = z;
    calView();
  }, { passive: false });
  stage.addEventListener('pointermove', e => {
    if (!CAL.shot) return;
    const [x, y] = toImg(e);
    CAL.coords.textContent = `${Math.round(x + CAL.shot.left)}, ${Math.round(y + CAL.shot.top)}`;
  });
  stage.addEventListener('pointerdown', e => {
    if (!CAL.shot || e.target.closest('button')) return;
    e.preventDefault();
    stage.setPointerCapture(e.pointerId);
    if (e.button === 1 || e.button === 2) {
      const [sx, sy, ox, oy] = [e.clientX, e.clientY, CAL.ox, CAL.oy];
      const move = ev => { CAL.ox = ox + ev.clientX - sx; CAL.oy = oy + ev.clientY - sy; calView(); };
      const up = () => { stage.removeEventListener('pointermove', move); stage.removeEventListener('pointerup', up); };
      stage.addEventListener('pointermove', move);
      stage.addEventListener('pointerup', up);
      return;
    }
    const tool = TOOLS.find(x => x.id === CAL.tool);
    const [x0, y0] = toImg(e);
    if (tool.kind === 'point') {
      commitPoint(x0, y0);
      return;
    }
    const draft = h('div', { class: 'shape draft', style: `--c: var(--${tool.c})` });
    CAL.layer.append(draft);
    let rect = null;
    const move = ev => {
      const [x1, y1] = toImg(ev);
      rect = { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
      Object.assign(draft.style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.w}px`, height: `${rect.h}px`, borderWidth: `${1.5 / CAL.zoom}px` });
    };
    const up = async () => {
      stage.removeEventListener('pointermove', move);
      stage.removeEventListener('pointerup', up);
      draft.remove();
      if (rect && rect.w >= 3 && rect.h >= 3) await commitRect(tool, rect);
    };
    stage.addEventListener('pointermove', move);
    stage.addEventListener('pointerup', up);
  });
}

async function commitRect(tool, rect) {
  await flush();
  if (tool.kind === 'region') {
    const r = await act('set_region', tool.name, rect);
    if (!r.ok) return;
    S.cfg = r.config;
    toast(t('toast.region'));
  } else {
    const r = await act('save_template', tool.name, rect);
    if (!r.ok) return;
    S.cfg = r.config;
    S.templates = r.templates;
    toast(t('toast.template'));
  }
  for (const path of binders.keys()) notify(path);
  CAL.renderTools();
  renderShapes();
}

async function commitPoint(x, y) {
  await flush();
  const r = await act('add_point', x, y);
  if (!r.ok) return;
  S.cfg = r.config;
  notify('cast.points');
  toast(t('toast.point'));
  CAL.renderTools();
  renderShapes();
}

// ── boot ────────────────────────────────────────────────────────────────────

function initChrome() {
  $$('[data-i]').forEach(el => el.replaceWith(ic(el.dataset.i)));
  $$('[data-menu]').forEach(b => b.addEventListener('click', () => openMenu(b, MENUS[b.dataset.menu]())));
  $$('[data-action="calibrate"]').forEach(b => b.addEventListener('click', () => openCalibrator()));
  $$('[data-action="check"]').forEach(b => b.addEventListener('click', showCheck));
  $$('[data-panel]').forEach(b => b.addEventListener('click', () => togglePanel(b.dataset.panel)));
  $('#logcard').addEventListener('click', () => togglePanel('log'));
  $('#run').addEventListener('click', async () => { await flush(); const r = await act('toggle'); if (r.ok) poll(); });
  $('#stop').addEventListener('click', () => act('stop'));
  $('#reset-stats').addEventListener('click', () => act('reset_stats'));
  $('#toggle-dock').addEventListener('click', () => {
    S.dock = !S.dock;
    document.body.classList.toggle('dock-min', !S.dock);
    if (!S.dock && S.panel) togglePanel(S.panel);
    saveUi();
  });
  $('#prev-profile').addEventListener('click', () => cycleProfile(-1));
  $('#next-profile').addEventListener('click', () => cycleProfile(1));
  $('#profile-tab').addEventListener('click', e => openMenu(e.currentTarget, S.boot.profiles.map(name =>
    ({ icon: name === S.boot.profile ? 'check' : 'folder', label: name, fn: () => switchProfile(name) }))));
  $('#lang').addEventListener('click', () => {
    lang = lang === 'ru' ? 'en' : 'ru';
    applyI18n();
    renderNodes();
    renderLogCard();
    if (S.snap) applySnap({ ...S.snap, logs: [] });
    if (S.panel) renderPanel();
    saveUi();
  });
  addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeMenu(); if (S.panel) togglePanel(S.panel); }
  });
  addEventListener('beforeunload', () => { if (Object.keys(pending).length) flush(); });
}

async function boot() {
  initChrome();
  initCanvas();
  const r = await call('bootstrap');
  if (!r.ok) { setTimeout(boot, 1500); return; }
  const ui = r.ui || {};
  lang = ui.lang || ((navigator.language || '').toLowerCase().startsWith('ru') ? 'ru' : 'en');
  S.layout = ui.layout || {};
  S.wires = ui.wires !== false;
  S.dock = ui.dock !== false;
  document.body.classList.toggle('no-wires', !S.wires);
  document.body.classList.toggle('dock-min', !S.dock);
  $('[data-zoom="wires"]').classList.toggle('on', S.wires);
  applyI18n();
  applyBoot(r);
  renderLogCard();
  if (ui.zoom && ui.pan) { S.zoom = ui.zoom; S.pan = ui.pan; applyView(); } else requestAnimationFrame(fit);
  poll();
  previewLoop();
  const firstRun = !Object.values(r.templates).some(Boolean) && !r.demo && !ui.layout;
  if (firstRun) setTimeout(showGuide, 500);
}

boot();
