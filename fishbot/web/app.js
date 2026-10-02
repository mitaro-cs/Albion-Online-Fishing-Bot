import { I18N } from './i18n.js';

// ── utilities ───────────────────────────────────────────────────────────────

const $ = (s, el = document) => el.querySelector(s);
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
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/>',
  check: '<path d="M5.5 12.5l4 4 9-9.5"/>',
  refresh: '<path d="M19.5 11.5a7.5 7.5 0 10-2.2 5.3"/><path d="M19.5 5v6.5H13"/>',
  folder: '<path d="M4 7.8a2 2 0 012-2h3.6l2 2.4H18a2 2 0 012 2v6.6a2 2 0 01-2 2H6a2 2 0 01-2-2z"/>',
  trash: '<path d="M5.5 7.5h13M10 7.5V5.5h4v2M7.5 7.5l.8 11h7.4l.8-11"/>',
  chart: '<path d="M6 18.5v-6M12 18.5v-12M18 18.5v-8.5"/>',
};

function ic(name) {
  const tpl = document.createElement('template');
  tpl.innerHTML = `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ''}</svg>`;
  return tpl.content.firstChild;
}

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
  document.querySelectorAll('[data-t]').forEach(el => { el.textContent = t(el.dataset.t); });
  $('#lang').textContent = t('lang');
}

// ── API ─────────────────────────────────────────────────────────────────────

const TOKEN = new URLSearchParams(location.search).get('token') || '';
let failures = 0;

async function call(method, ...args) {
  if (S.gone) return { ok: false, error: 'gone' };
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

// ── state & two-way binding ─────────────────────────────────────────────────

const S = { boot: null, cfg: null, schema: null, templates: {}, snap: null, logs: [], lastLog: 0, catches: [] };

const getPath = path => path.split('.').reduce((o, k) => (o == null ? o : o[k]), S.cfg);
const meta = path => path.split('.').reduce((o, k) => (o == null ? o : o[k]), S.schema) || {};
const binders = new Map();

function bind(path, fn) {
  if (!binders.has(path)) binders.set(path, []);
  binders.get(path).push(fn);
  fn(getPath(path));
}

function notify(path) {
  for (const fn of binders.get(path) || []) fn(getPath(path));
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
  if (S.snap) paint(S.snap);
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
  const v = parseFloat(String(raw).replace(',', '.'));
  if (!Number.isFinite(v)) return getPath(path);
  const c = clamp(v, m.min ?? -Infinity, m.max ?? Infinity);
  return m.type === 'int' ? Math.round(c) : +c.toFixed(4);
}

function row(path, control, cls = '') {
  const label = t(`f.${path}`);
  return h('div', { class: `row ${cls}` }, h('label', { title: label }, label), control);
}

function stepper(path) {
  const step = meta(path).step || 1;
  const input = h('input', { class: 'ctl', inputmode: 'decimal', spellcheck: 'false' });
  const show = () => { input.value = fmt(path, getPath(path)); };
  input.addEventListener('focus', () => { input.value = getPath(path); input.select(); });
  input.addEventListener('blur', () => { setValue(path, coerce(path, input.value)); show(); });
  input.addEventListener('keydown', e => { if (e.key === 'Enter') input.blur(); });
  const nudge = dir => setValue(path, coerce(path, getPath(path) + dir * step));
  const btn = (cls, dir) => {
    const b = h('button', { class: cls, type: 'button', tabindex: '-1', 'aria-label': cls });
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
  return row(path, h('div', { class: 'stepper' }, btn('dec', -1), input, btn('inc', 1)));
}

function slider(path) {
  const m = meta(path);
  const input = h('input', { type: 'range', min: m.min, max: m.max, step: m.step || 1 });
  const val = h('span', { class: 'val' });
  const paintVal = v => {
    input.style.setProperty('--p', `${((v - m.min) / (m.max - m.min)) * 100}%`);
    val.textContent = fmt(path, v);
  };
  input.addEventListener('input', () => { paintVal(+input.value); setValue(path, coerce(path, input.value)); });
  bind(path, v => { input.value = v; paintVal(v); });
  return row(path, h('div', { class: 'slider' }, input, val), 'slide');
}

function toggle(path) {
  const el = h('button', { class: 'toggle', type: 'button', role: 'switch', onclick: () => setValue(path, !getPath(path)) });
  bind(path, v => { el.classList.toggle('on', !!v); el.setAttribute('aria-checked', String(!!v)); });
  return row(path, el, 'check');
}

function select(path, options) {
  const el = h('select', { class: 'ctl', onchange: e => setValue(path, Number(e.target.value)) },
    options.map(([v, label]) => h('option', { value: v }, label)));
  bind(path, v => { el.value = v; });
  return row(path, el);
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
      removeEventListener('keydown', onKey, true);
      listening = false;
      const name = e.key === 'Escape' ? null : keyName(e);
      if (name) setValue(path, name);
      show();
    };
    addEventListener('keydown', onKey, true);
  });
  bind(path, show);
  return row(path, el);
}

// ── screen ──────────────────────────────────────────────────────────────────

let els = {};

function render() {
  binders.clear();
  els = {};
  $('#status').replaceChildren(...buildStatus());
  $('#settings').replaceChildren(buildSettings());
  $('#vision').replaceChildren(...buildVision());
  renderLog();
  if (S.snap) paint(S.snap);
}

function buildStatus() {
  els.stage = h('div', { class: 'st-stage' });
  els.sub = h('div', { class: 'st-sub' });
  els.count = h('div', { class: 'st-count' });
  els.rate = h('div', { class: 'st-rate' });
  els.run = h('button', { class: 'btn primary', type: 'button', onclick: toggleRun });
  els.learn = h('div', { class: 'learn' });
  els.stats = h('div', { class: 'stats' });
  els.log = h('div', { class: 'log' });
  ['regions', 'bite.use_sound', 'bite.sound_prints', 'bite.method', 'reel.method'].forEach(p => bind(p, paintLearn));
  return [
    h('div', { class: 'st-head' },
      h('div', {}, els.stage, els.sub),
      h('div', { class: 'st-score' }, els.count, els.rate)),
    h('div', { class: 'st-run' }, els.run,
      h('button', { class: 'btn icon', type: 'button', title: t('btn.stop'), 'aria-label': t('btn.stop'), onclick: () => act('stop') }, ic('stop'))),
    els.learn,
    els.stats,
    h('h3', {}, t('log')),
    els.log,
    h('div', { class: 'st-foot' },
      h('button', { class: 'btn sm', type: 'button', title: t('act.forget.tip'), onclick: forget }, ic('refresh'), t('act.forget')),
      h('button', { class: 'btn sm', type: 'button', onclick: () => act('open_debug') }, ic('folder'), t('act.debug')),
      h('button', { class: 'btn sm', type: 'button', onclick: () => act('reset_stats') }, ic('chart'), t('act.reset_stats')),
      h('button', { class: 'btn sm danger', type: 'button', onclick: uninstallApp }, ic('trash'), t('act.uninstall'))),
  ];
}

function buildSettings() {
  const group = (key, color, ...rows) => h('div', { class: 'group' },
    h('h3', {}, h('span', { class: 'dot', style: `--c: var(--${color})` }), t(key)), rows);
  const monitors = S.boot.monitors > 1
    ? select('system.monitor', Array.from({ length: S.boot.monitors }, (_, i) => [i + 1, String(i + 1)])) : null;
  return h('div', { class: 'groups' },
    group('g.fish', 'yellow',
      slider('cast.power_ms'),
      stepper('session.cooldown_ms'),
      stepper('bite.bite_timeout_s'),
      stepper('bite.hook_delay_ms')),
    group('g.bite', 'green',
      toggle('bite.use_sound'),
      stepper('bite.sound_sensitivity'),
      stepper('bite.confirm_ms')),
    group('g.session', 'blue',
      stepper('session.max_catches'),
      stepper('session.max_minutes'),
      stepper('session.break_every_min'),
      stepper('session.break_minutes')),
    group('g.system', 'pink',
      keyInput('system.hotkey_toggle'),
      keyInput('system.hotkey_stop'),
      toggle('system.require_focus'),
      toggle('system.auto_update'),
      toggle('system.sound'),
      monitors));
}

function buildVision() {
  const img = h('img', { alt: '' });
  const backdrop = h('img', { alt: '', class: 'backdrop' });
  const badge = h('div', { class: 'badge' }, h('span', { class: 'live' }), h('span', {}, t('vision.live')));
  const title = h('h4');
  const text = h('p');
  const frame = h('div', { class: 'frame' }, h('div', { class: 'aurora' }), backdrop, img, badge, h('div', { class: 'caption' }, title, text));
  img.hidden = backdrop.hidden = badge.hidden = true;
  els.vision = { img, backdrop, badge, title, text, frame };
  idleCaption();
  return [h('h3', {}, h('span', { class: 'dot', style: '--c: var(--blue)' }), t('vision')), frame];
}

function idleCaption() {
  const v = els.vision;
  if (!v || v.frame.classList.contains('has-image')) return;
  v.title.textContent = t('vision.idle');
  v.text.textContent = t('vision.hint', { key: S.cfg.system.hotkey_toggle.toUpperCase() });
}

function paintLearn() {
  const c = S.cfg, tp = S.templates;
  if (!els.learn || !c) return;
  const bob = c.regions.bobber.width >= 4 && (c.bite.method === 'color' || !!tp.bobber);
  const bar = c.regions.reel.width >= 4 && (['bar', 'color'].includes(c.reel.method) || !!tp.marker);
  const prints = c.bite.sound_prints.length;
  const heard = S.snap ? S.snap.sound : S.boot.sound;
  const line = (state, label, detail, extra) => h('div', { class: `l ${state}` }, h('i'), h('span', {}, label), h('b', {}, detail, extra));
  const sound = !c.bite.use_sound ? line('off', t('learn.sound'), t('learn.off'))
    : prints >= 2 ? line('ok', t('learn.sound'), t('learn.done'),
      h('button', { class: 'link', type: 'button', onclick: forgetSound }, t('learn.reset')))
    : line('', t('learn.sound'), heard ? t('learn.sound_n', { n: prints }) : t('learn.nosound'));
  els.learn.replaceChildren(
    line(bob ? 'ok' : '', t('learn.bobber'), t(bob ? 'learn.done' : 'learn.first_cast')),
    line(bar ? 'ok' : '', t('learn.bar'), t(bar ? 'learn.done' : 'learn.first_fish')),
    sound);
  S.learnKey = learnKey();
}

function learnKey() {
  const c = S.cfg;
  return [c.regions.bobber.width, c.regions.reel.width, c.bite.method, c.reel.method, c.bite.sound_prints.length,
    c.bite.use_sound, S.snap ? S.snap.sound : S.boot.sound, !!S.templates.bobber, !!S.templates.marker].join('|');
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
  if (p.way) p.way = t(`way.${p.way}`);
  if (e.code === 'loot' && p.name === '?') p.name = t('loot.unnamed');
  return t(`ev.${e.code}`, p);
}

function stageLabel(snap) {
  if (snap.status === 'paused') return snap.pause_reason === 'focus' ? t('paused.focus') : t('stage.paused');
  return t(`stage.${snap.stage}`);
}

// change text and replay a one-shot CSS animation, but not on the first paint
function swapText(el, text, cls) {
  if (el.textContent === text) return;
  const first = !el.textContent;
  el.textContent = text;
  if (first) return;
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
}

function paint(snap) {
  const s = snap.stats;
  document.body.classList.toggle('running', snap.status === 'running');
  document.body.classList.toggle('paused', snap.status === 'paused');
  swapText(els.stage, stageLabel(snap), 'swap');
  els.sub.textContent = snap.remaining ? `${Math.ceil(snap.remaining)} ${UNITS[lang].s}` : '';
  swapText(els.count, String(s.catches), 'bump');
  els.rate.textContent = t('rate', { n: s.per_hour });
  const runKey = snap.status + lang + S.cfg.system.hotkey_toggle;
  if (els.run.dataset.state !== runKey) {
    els.run.dataset.state = runKey;
    els.run.replaceChildren(ic(snap.status === 'running' ? 'pause' : 'play'),
      t(snap.status === 'running' ? 'btn.pause' : snap.status === 'paused' ? 'btn.resume' : 'btn.start'),
      h('kbd', {}, S.cfg.system.hotkey_toggle.toUpperCase()));
  }
  const cards = [['casts', s.casts], ['bites', s.bites], ['escaped', s.escaped],
    ['misses', s.misses], ['success', `${s.success}%`], ['active', fmtClock(s.active_s)]];
  els.stats.replaceChildren(...cards.map(([k, v]) => h('div', { class: 'stat' }, h('div', { class: 'v' }, v), h('div', { class: 'k' }, t(`stat.${k}`)))));
  if (learnKey() !== S.learnKey) paintLearn();
}

function applySnap(snap) {
  if (S.gone) return;
  S.snap = snap;
  if (snap.logs.length) {
    S.logs.push(...snap.logs);
    if (S.logs.length > 200) S.logs.splice(0, S.logs.length - 200);
    S.lastLog = snap.logs.at(-1).id;
    appendLog(snap.logs);
    for (const e of snap.logs) if (e.level === 'error') toast(eventText(e), 'error');
  }
  paint(snap);
  if (snap.catch_rev !== S.catchRev) { S.catchRev = snap.catch_rev; loadCatches(); }
}

function logRow(e) {
  const time = new Date(e.ts * 1000).toLocaleTimeString(lang, { hour12: false });
  return h('div', { class: `e ${e.level}` }, h('time', {}, time), h('i'), h('span', { title: eventText(e) }, eventText(e)));
}

function renderLog() {
  els.log.replaceChildren(...(S.logs.length ? S.logs.slice(-30).map(logRow) : [h('div', { class: 'e' }, h('time'), h('i'), h('span', {}, t('log.empty')))]));
}

function appendLog(events) {
  if (!els.log) return;
  if (!els.log.querySelector('time:not(:empty)')) els.log.replaceChildren();
  els.log.append(...events.map(logRow));
  while (els.log.childElementCount > 30) els.log.firstChild.remove();
}

let polling = false;

async function poll() {
  if (polling) return;
  polling = true;
  const r = await call('state', S.lastLog);
  polling = false;
  if (r.ok) applySnap(r);
  if (r.ok && r.cfg_rev !== S.cfgRev) {  // config changed on the backend (e.g. the bot learned something)
    const first = S.cfgRev === undefined;
    S.cfgRev = r.cfg_rev;
    if (!first && !Object.keys(pending).length) {
      const fresh = await call('setup_state');
      if (fresh.ok) { S.templates = fresh.templates; applyConfig(fresh.config); paintLearn(); }
    }
  }
  const busy = r.ok && r.status !== 'idle';
  setTimeout(poll, !r.ok ? 1500 : r.stage === 'reel' ? 80 : busy ? 200 : 700);
}

async function previewLoop() {
  const snap = S.snap;
  if (snap && snap.status === 'running' && !document.hidden) {
    const r = await call('preview');
    if (r.ok && r.image) showFrame(r.image, stageLabel(snap), S.logs.length ? eventText(S.logs.at(-1)) : '');
  }
  setTimeout(previewLoop, 110);
}

function showFrame(src, title, text) {
  const v = els.vision;
  if (!v) return;
  v.img.src = v.backdrop.src = src;
  v.img.hidden = v.backdrop.hidden = v.badge.hidden = false;
  v.frame.classList.add('has-image');
  v.title.textContent = title;
  v.text.textContent = text;
}

async function toggleRun() {
  await flush();
  const r = await act('toggle');
  if (r.ok) poll();
}

async function forget() {
  const r = await act('forget');
  if (r.ok) { S.templates = r.templates; applyConfig(r.config); paintLearn(); toast(t('toast.forgot')); }
}

async function forgetSound() {
  const r = await act('forget_sound');
  if (r.ok) { applyConfig(r.config); paintLearn(); toast(t('toast.sound_reset')); }
}

// ── catch summary (read from the game's loot banner) ────────────────────────

async function loadCatches() {
  const r = await call('catches');
  if (!r.ok) return;
  S.catches = r.items;
  const bar = $('#loot');
  bar.hidden = !r.items.length;
  const top = [...r.items].sort((a, b) => b.count - a.count).slice(0, 4);
  bar.replaceChildren(...[h('span', { class: 'k' }, t('loot.title')), ...top.map(lootChip),
    r.items.length > 4 && h('span', { class: 'more-n' }, `+${r.items.length - 4}`)].filter(Boolean));
}

function lootChip(it) {
  return h('span', { class: 'lchip' },
    it.name ? h('span', {}, it.name) : h('img', { src: it.image, alt: '' }),
    h('b', {}, `×${it.count}`));
}

// ── toasts, dialogs, uninstall, self-update ─────────────────────────────────

function toast(text, kind = 'ok') {
  const el = h('div', { class: `toast ${kind}` }, h('i'), h('span', {}, text));
  $('#toasts').append(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 320); }, kind === 'error' ? 4200 : 2400);
}

function dialog(titleKey, body, buttons = [['btn.ok', true]]) {
  return new Promise(resolve => {
    const close = v => { modal.remove(); removeEventListener('keydown', onKey, true); resolve(v); };
    const onKey = e => { if (e.key === 'Escape') { e.stopPropagation(); close(null); } };
    const modal = h('div', { class: 'modal dialog', onpointerdown: e => { if (e.target === modal) close(null); } },
      h('div', { class: 'box' }, h('h3', {}, t(titleKey)), body,
        h('div', { class: 'actions' }, buttons.map(([k, v, primary]) =>
          h('button', { class: `btn sm${primary ? ' primary' : ''}`, onclick: () => close(v) }, t(k))))));
    document.body.append(modal);
    addEventListener('keydown', onKey, true);
  });
}

async function uninstallApp() {
  const ok = await dialog('uninstall.title', h('div', {},
    h('p', {}, t('uninstall.text')),
    h('div', { class: 'note' }, S.boot.data_dir)), [['btn.cancel', false], ['uninstall.confirm', true, true]]);
  if (!ok) return;
  const r = await act('uninstall');
  if (!r.ok) return;
  S.gone = true;
  document.body.replaceChildren(h('div', { class: 'gone' },
    h('div', { class: 'mark' }, ic('check')), h('h2', {}, t('uninstall.done')), h('p', {}, t('uninstall.done2'))));
}

async function updateLoop() {
  const r = await call('update_state');
  const u = r.ok ? r.update : null;
  const chip = $('#update-chip');
  if (u && (u.status === 'downloading' || u.status === 'ready')) {
    chip.hidden = false;
    chip.classList.toggle('ready', u.status === 'ready');
    chip.replaceChildren(h('i'), u.status === 'ready'
      ? t('update.ready', { v: u.latest })
      : t('update.downloading', { v: u.latest, p: Math.round(u.progress * 100) }));
    chip.onclick = u.status === 'ready' ? async () => {
      const ok = await dialog('update.title', h('p', {}, t('update.confirm', { v: u.latest })),
        [['btn.cancel', false], ['update.now', true, true]]);
      if (ok && (await act('update_install')).ok) toast(t('update.restarting'));
    } : null;
  }
  if (!u || ['idle', 'checking', 'downloading'].includes(u.status)) setTimeout(updateLoop, 2000);
}

// ── boot ────────────────────────────────────────────────────────────────────

async function boot() {
  const r = await call('bootstrap');
  if (!r.ok) { setTimeout(boot, 1500); return; }
  S.boot = r;
  S.schema = r.schema;
  S.cfg = r.config;
  S.templates = r.templates;
  lang = r.ui?.lang || ((navigator.language || '').toLowerCase().startsWith('ru') ? 'ru' : 'en');
  $('#version').textContent = r.version;
  $('#demo-tag').hidden = !r.demo;
  $('#lang').addEventListener('click', () => {
    lang = lang === 'ru' ? 'en' : 'ru';
    applyI18n();
    render();
    loadCatches();
    call('save_ui', { lang });
  });
  addEventListener('beforeunload', () => { if (Object.keys(pending).length) flush(); });
  applyI18n();
  render();
  poll();
  previewLoop();
  updateLoop();
}

boot();
