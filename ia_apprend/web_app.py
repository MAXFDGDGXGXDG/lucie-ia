from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

try:
    from .ai_bot import LearningBot
except ImportError:  # pragma: no cover - script fallback
    from ai_bot import LearningBot


HTML_PAGE = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lucie</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #050814;
      --panel: #0f172a;
      --panel-2: #0b1220;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #22c55e;
      --border: rgba(148, 163, 184, 0.16);
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background:
        radial-gradient(circle at top, rgba(56, 189, 248, 0.09), transparent 32%),
        linear-gradient(180deg, #07111f 0%, var(--bg) 100%);
      color: var(--text);
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 20px;
      padding: 22px;
    }
    .topbar {
      width: min(1100px, 100%);
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .brand {
      display: grid;
      gap: 4px;
    }
    .brand h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }
    .brand p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.68);
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--accent-2);
      box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.14);
    }
    .stage,
    .chat-shell {
      width: min(1100px, 100%);
      margin: 0 auto;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.9fr);
      gap: 18px;
      align-items: start;
    }
    .chat-card {
      min-height: min(68vh, 760px);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 16px;
      padding: 18px;
      border-radius: 24px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.78);
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.26);
    }
    .brain-card {
      position: sticky;
      top: 18px;
      display: grid;
      gap: 14px;
      padding: 18px;
      border-radius: 24px;
      border: 1px solid var(--border);
      background:
        radial-gradient(circle at top, rgba(56, 189, 248, 0.16), transparent 42%),
        linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.92));
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.24);
      overflow: hidden;
    }
    .brain-orb {
      width: 100%;
      aspect-ratio: 1.25;
      border-radius: 28px;
      border: 1px solid rgba(56, 189, 248, 0.18);
      background:
        radial-gradient(circle at 50% 42%, rgba(56, 189, 248, 0.34), transparent 28%),
        radial-gradient(circle at 50% 52%, rgba(34, 197, 94, 0.18), transparent 34%),
        radial-gradient(circle at 50% 55%, rgba(148, 163, 184, 0.14), transparent 48%),
        linear-gradient(180deg, rgba(2, 6, 23, 0.5), rgba(15, 23, 42, 0.85));
      box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.02),
        0 0 0 1px rgba(56, 189, 248, 0.08),
        0 18px 30px rgba(0, 0, 0, 0.22);
      position: relative;
      overflow: hidden;
    }
    .brain-orb::before,
    .brain-orb::after {
      content: "";
      position: absolute;
      inset: 18% 14%;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      animation: drift 10s linear infinite;
    }
    .brain-orb::after {
      inset: 28% 24%;
      animation-duration: 14s;
      animation-direction: reverse;
    }
    .brain-core {
      position: absolute;
      inset: 33% 33%;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(255,255,255,0.9), rgba(56,189,248,0.54) 26%, rgba(34,197,94,0.18) 58%, transparent 70%);
      filter: blur(2px);
      animation: pulse 4.5s ease-in-out infinite;
    }
    .brain-node {
      position: absolute;
      width: 11px;
      height: 11px;
      border-radius: 999px;
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(56,189,248,0.72));
      box-shadow: 0 0 18px rgba(56, 189, 248, 0.75);
      animation: float 5s ease-in-out infinite;
    }
    .brain-node.node-a { top: 18%; left: 18%; }
    .brain-node.node-b { top: 18%; right: 18%; animation-delay: 0.7s; }
    .brain-node.node-c { bottom: 22%; left: 24%; animation-delay: 1.3s; }
    .brain-node.node-d { bottom: 18%; right: 24%; animation-delay: 1.8s; }
    .brain-node.node-e { top: 46%; left: 9%; animation-delay: 2.2s; }
    .brain-node.node-f { top: 46%; right: 9%; animation-delay: 2.8s; }
    .brain-panel {
      display: grid;
      gap: 10px;
    }
    .brain-title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }
    .brain-title strong {
      font-size: 16px;
    }
    .brain-title span {
      color: var(--muted);
      font-size: 12px;
    }
    .brain-focus {
      padding: 14px;
      border-radius: 18px;
      border: 1px solid rgba(56, 189, 248, 0.16);
      background: rgba(2, 6, 23, 0.46);
      display: grid;
      gap: 6px;
    }
    .brain-focus .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .brain-focus .value {
      font-size: 18px;
      font-weight: 700;
      line-height: 1.35;
    }
    .brain-focus .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .brain-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .brain-stat {
      padding: 12px;
      border-radius: 16px;
      background: rgba(2, 6, 23, 0.42);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .brain-stat .k {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .brain-stat .v {
      font-size: 18px;
      font-weight: 700;
    }
    .brain-feed {
      display: grid;
      gap: 10px;
      max-height: 280px;
      overflow: auto;
      padding-right: 2px;
    }
    .brain-chip {
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.14);
      background: rgba(2, 6, 23, 0.38);
      font-size: 13px;
      line-height: 1.45;
    }
    .brain-chip strong {
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
    }
    .brain-empty {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px dashed rgba(148, 163, 184, 0.16);
      background: rgba(2, 6, 23, 0.24);
    }
    @keyframes pulse {
      0%, 100% { transform: scale(0.98); opacity: 0.88; }
      50% { transform: scale(1.04); opacity: 1; }
    }
    @keyframes float {
      0%, 100% { transform: translateY(0px); opacity: 0.82; }
      50% { transform: translateY(-6px); opacity: 1; }
    }
    @keyframes drift {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    .chat-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }
    .chat-copy {
      display: grid;
      gap: 4px;
    }
    .chat-copy strong {
      font-size: 16px;
    }
    .chat-copy span {
      color: var(--muted);
      font-size: 13px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .chips button {
      border: 1px solid var(--border);
      background: rgba(2, 6, 23, 0.55);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }
    .chips button:hover {
      transform: translateY(-1px);
      border-color: rgba(56, 189, 248, 0.36);
      background: rgba(2, 6, 23, 0.8);
    }
    .chat-log {
      display: grid;
      align-content: start;
      gap: 12px;
      overflow: auto;
      padding-right: 4px;
      min-height: 320px;
    }
    .bubble {
      max-width: min(760px, 100%);
      padding: 14px 16px;
      border-radius: 18px;
      line-height: 1.55;
      white-space: pre-wrap;
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.12);
    }
    .bubble.user {
      justify-self: end;
      background: linear-gradient(180deg, rgba(56, 189, 248, 0.18), rgba(14, 165, 233, 0.11));
      border-color: rgba(56, 189, 248, 0.24);
    }
    .bubble.assistant {
      justify-self: start;
      background: rgba(2, 6, 23, 0.48);
    }
    .bubble.system {
      justify-self: center;
      color: var(--muted);
      background: transparent;
      border-style: dashed;
      box-shadow: none;
      text-align: center;
    }
    .composer {
      display: grid;
      gap: 12px;
    }
    textarea {
      width: 100%;
      min-height: 110px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(2, 6, 23, 0.72);
      color: var(--text);
      padding: 16px 18px;
      font: inherit;
      line-height: 1.5;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.14);
    }
    .actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .actions button[type="submit"] {
      border: 1px solid rgba(56, 189, 248, 0.28);
      background: linear-gradient(180deg, rgba(56, 189, 248, 0.22), rgba(14, 165, 233, 0.16));
      color: var(--text);
      border-radius: 999px;
      padding: 11px 18px;
      cursor: pointer;
      font-weight: 600;
    }
    .actions button[type="submit"]:hover {
      border-color: rgba(56, 189, 248, 0.42);
    }
    .hint {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .mini {
      color: var(--muted);
      font-size: 13px;
    }
    .hidden-btn {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    @media (max-width: 700px) {
      .shell { padding: 16px; }
      .topbar { flex-direction: column; align-items: flex-start; }
      .workspace { grid-template-columns: 1fr; }
      .brain-card { position: static; }
      .chat-card { min-height: 70vh; padding: 14px; }
      .bubble { max-width: 100%; }
      textarea { min-height: 96px; font-size: 16px; }
    }

    /* Interface blanche, plus proche d'un assistant moderne. */
    :root {
      color-scheme: light;
      --bg: #f7f7f8;
      --panel: #ffffff;
      --panel-2: #f3f4f6;
      --text: #202123;
      --muted: #6b7280;
      --accent: #10a37f;
      --accent-2: #10a37f;
      --border: #e5e7eb;
      --soft: #f4f4f5;
    }
    body {
      background: var(--bg);
      color: var(--text);
    }
    .shell {
      min-height: 100vh;
      padding: 0;
      gap: 0;
      grid-template-rows: auto 1fr;
    }
    .topbar {
      width: 100%;
      min-height: 64px;
      padding: 0 24px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .brand h1 {
      font-size: 18px;
      font-weight: 700;
    }
    .brand p {
      display: none;
    }
    .status {
      background: #ffffff;
      color: #374151;
      border-color: var(--border);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
      font-size: 13px;
      padding: 8px 12px;
    }
    .dot {
      width: 8px;
      height: 8px;
      box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.14);
    }
    .stage {
      width: 100%;
      margin: 0;
    }
    .workspace {
      width: min(1220px, 100%);
      min-height: calc(100vh - 64px);
      margin: 0 auto;
      grid-template-columns: 1fr;
      gap: 0;
      align-items: stretch;
    }
    .chat-card {
      order: 2;
      min-height: calc(100vh - 64px);
      padding: 0;
      gap: 0;
      border: 0;
      border-left: 1px solid var(--border);
      border-radius: 0;
      background: #ffffff;
      box-shadow: none;
      grid-template-rows: auto 1fr auto;
    }
    .brain-card {
      display: none;
    }
    .brain-orb {
      display: none;
    }
    .brain-panel {
      gap: 14px;
    }
    .brain-title {
      padding: 2px 2px 8px;
    }
    .brain-title strong,
    .chat-copy strong {
      color: #111827;
      font-size: 15px;
      font-weight: 700;
    }
    .brain-title span,
    .chat-copy span,
    .mini,
    .hint {
      color: var(--muted);
    }
    .brain-focus,
    .brain-stat,
    .brain-chip,
    .brain-empty {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .brain-focus .label,
    .brain-stat .k {
      color: #6b7280;
      letter-spacing: 0.04em;
    }
    .brain-focus .value,
    .brain-stat .v {
      color: #111827;
      font-size: 17px;
    }
    .chat-head {
      width: min(820px, 100%);
      margin: 0 auto;
      padding: 28px 24px 8px;
      align-items: flex-start;
    }
    .chat-copy {
      gap: 6px;
    }
    .chips {
      display: none;
    }
    .chips button {
      background: #ffffff;
      color: #374151;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .chips button:hover {
      transform: translateY(-1px);
      background: #f9fafb;
      border-color: #d1d5db;
    }
    .chat-log {
      width: min(820px, 100%);
      margin: 0 auto;
      padding: 20px 24px 28px;
      gap: 18px;
      min-height: 360px;
    }
    .bubble {
      max-width: min(680px, 100%);
      border: 0;
      box-shadow: none;
      color: #202123;
      font-size: 15px;
      line-height: 1.65;
      padding: 0;
    }
    .bubble.user {
      justify-self: end;
      background: #f4f4f4;
      border-radius: 20px;
      padding: 12px 16px;
    }
    .bubble.assistant {
      justify-self: start;
      background: transparent;
      padding-left: 44px;
      position: relative;
    }
    .bubble.assistant::before {
      content: "L";
      position: absolute;
      left: 0;
      top: 2px;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      background: #10a37f;
      color: #ffffff;
      font-size: 13px;
      font-weight: 700;
    }
    .bubble-text {
      white-space: pre-wrap;
    }
    .bubble-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .bubble-actions button,
    .confidence-pill {
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #fff;
      color: #374151;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
    }
    .bubble-actions button {
      cursor: pointer;
    }
    .confidence-pill {
      border-color: #d1fae5;
      color: #047857;
      background: #ecfdf5;
    }
    .bubble.system {
      width: 100%;
      color: #6b7280;
      background: #f9fafb;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
    }
    .composer {
      position: sticky;
      bottom: 0;
      padding: 14px 24px 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.78), #ffffff 24%);
      border-top: 1px solid rgba(229, 231, 235, 0.72);
    }
    .composer form,
    .hint {
      width: min(820px, 100%);
      margin-left: auto;
      margin-right: auto;
    }
    .composer form {
      display: grid;
      gap: 10px;
    }
    textarea {
      min-height: 58px;
      max-height: 180px;
      resize: vertical;
      border-radius: 24px;
      background: #ffffff;
      color: #111827;
      border: 1px solid #d1d5db;
      padding: 16px 18px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
    }
    textarea:focus {
      border-color: #10a37f;
      box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.12), 0 8px 30px rgba(15, 23, 42, 0.08);
    }
    .actions {
      align-items: center;
    }
    .actions button[type="submit"] {
      background: #111827;
      color: #ffffff;
      border-color: #111827;
      border-radius: 999px;
      padding: 10px 16px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12);
    }
    .actions button[type="submit"]:hover {
      background: #000000;
      border-color: #000000;
    }
    .onboarding {
      position: fixed;
      inset: 0;
      z-index: 80;
      display: grid;
      place-items: center;
      padding: 22px;
      background: rgba(249, 250, 251, 0.82);
      backdrop-filter: blur(10px);
    }
    .onboarding[hidden] {
      display: none;
    }
    .onboarding-card {
      width: min(620px, 100%);
      display: grid;
      gap: 16px;
      padding: 24px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: #ffffff;
      box-shadow: 0 24px 70px rgba(15, 23, 42, 0.18);
      color: #111827;
    }
    .onboarding-card h2 {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }
    .onboarding-card p {
      margin: 0;
      color: #6b7280;
      line-height: 1.5;
    }
    .onboarding-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .onboarding-field {
      display: grid;
      gap: 6px;
    }
    .onboarding-field.full {
      grid-column: 1 / -1;
    }
    .onboarding-field label {
      font-size: 13px;
      font-weight: 700;
      color: #374151;
    }
    .onboarding-field input,
    .onboarding-field select {
      width: 100%;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      padding: 12px 13px;
      font: inherit;
      color: #111827;
      background: #ffffff;
    }
    .onboarding-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      flex-wrap: wrap;
    }
    .onboarding-actions button {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 10px 16px;
      font-weight: 800;
      cursor: pointer;
      background: #ffffff;
      color: #374151;
    }
    .onboarding-actions button[type="submit"] {
      background: #111827;
      color: #ffffff;
      border-color: #111827;
    }
    code {
      color: #374151;
      background: #f3f4f6;
      border-radius: 6px;
      padding: 2px 5px;
    }
    @media (max-width: 850px) {
      .topbar {
        position: static;
        min-height: auto;
        padding: 16px;
      }
      .workspace {
        grid-template-columns: 1fr;
        min-height: auto;
      }
      .brain-card {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }
      .chat-card {
        border-left: 0;
        min-height: 72vh;
      }
      .chat-head,
      .chat-log,
      .composer {
        padding-left: 16px;
        padding-right: 16px;
      }
      .onboarding-grid {
        grid-template-columns: 1fr;
      }
      .bubble.assistant {
        padding-left: 38px;
      }
    }

    .topbar {
      display: grid;
      grid-template-columns: auto 1fr auto;
    }
    .sidebar-toggle {
      width: 38px;
      height: 38px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #ffffff;
      color: #111827;
      font-size: 20px;
      cursor: pointer;
    }
    .sidebar-toggle:hover {
      background: #f9fafb;
    }
    .workspace {
      width: 100%;
      max-width: none;
      display: block;
    }
    .chat-sidebar {
      position: fixed;
      top: 64px;
      left: 0;
      z-index: 30;
      width: min(320px, calc(100vw - 20px));
      height: calc(100vh - 64px);
      border-right: 1px solid var(--border);
      background: #f7f7f8;
      padding: 14px;
      overflow: auto;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
      box-shadow: 18px 0 38px rgba(15, 23, 42, 0.14);
      transform: translateX(0);
      transition: transform 180ms ease, box-shadow 180ms ease, visibility 180ms ease;
    }
    body.sidebar-closed .chat-sidebar {
      transform: translateX(-105%);
      visibility: hidden;
      pointer-events: none;
      box-shadow: none;
    }
    .sidebar-head {
      display: grid;
      gap: 10px;
    }
    .sidebar-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .sidebar-line strong {
      font-size: 15px;
    }
    .sidebar-close {
      border: 0;
      background: transparent;
      color: #6b7280;
      font-size: 24px;
      cursor: pointer;
      line-height: 1;
    }
    .new-chat {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
      color: #111827;
      padding: 11px 12px;
      font-weight: 700;
      text-align: left;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .new-chat:hover {
      background: #f9fafb;
    }
    .report-btn {
      width: 100%;
      border: 1px solid #fecaca;
      border-radius: 12px;
      background: #fff7f7;
      color: #991b1b;
      padding: 11px 12px;
      font-weight: 700;
      text-align: left;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .report-btn:hover {
      background: #fee2e2;
    }
    .report-note {
      color: #6b7280;
      font-size: 12px;
      line-height: 1.35;
    }
    .mail-panel {
      display: grid;
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
    }
    .mail-panel strong {
      font-size: 13px;
    }
    .mail-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .mail-actions button {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #f9fafb;
      color: #111827;
      padding: 9px 10px;
      font-weight: 700;
      cursor: pointer;
    }
    .mail-actions button:hover {
      background: #f3f4f6;
    }
    .mail-status {
      color: #6b7280;
      font-size: 12px;
      line-height: 1.35;
    }
    .chat-list {
      display: grid;
      align-content: start;
      gap: 6px;
    }
    .chat-item {
      border: 0;
      border-radius: 10px;
      background: transparent;
      color: #374151;
      padding: 10px 11px;
      text-align: left;
      cursor: pointer;
      min-height: 38px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .chat-item:hover,
    .chat-item.active {
      background: #ececf1;
      color: #111827;
    }
    .chat-card {
      border-left: 0;
      width: min(940px, 100%);
      margin: 0 auto;
    }
    .chat-head {
      padding-top: 36px;
    }
    .chips,
    .hint,
    .mini {
      display: none;
    }
    .actions {
      justify-content: flex-end;
    }
    @media (max-width: 850px) {
      .topbar {
        grid-template-columns: auto 1fr;
      }
      .status {
        grid-column: 1 / -1;
      }
      .workspace {
        grid-template-columns: 1fr;
      }
      .chat-sidebar {
        top: 64px;
        height: calc(100vh - 64px);
        border-right: 1px solid var(--border);
        border-bottom: 0;
      }
    }
  </style>
</head>
<body class="sidebar-closed">
  <div class="shell">
    <header class="topbar">
      <button class="sidebar-toggle" id="sidebar-toggle" type="button" aria-label="Ouvrir ou fermer les discussions" onclick="document.body.classList.toggle('sidebar-closed')">☰</button>
      <div class="brand">
        <h1>Lucie</h1>
        <p>Assistant personnel</p>
      </div>
      <div id="status" class="status"><span class="dot"></span><span>En attente...</span></div>
    </header>

    <main class="stage">
      <div class="workspace">
        <aside class="chat-sidebar">
          <div class="sidebar-head">
            <div class="sidebar-line">
              <strong>Discussions</strong>
              <button class="sidebar-close" id="sidebar-close" type="button" aria-label="Fermer" onclick="document.body.classList.add('sidebar-closed')">×</button>
            </div>
            <button class="new-chat" id="new-chat" type="button">+ Nouveau chat</button>
            <button class="report-btn" id="report-bad-answer" type="button">Signaler la derniere reponse</button>
            <a class="report-note" href="/admin">Admin Maxence</a>
            <div class="report-note">Envoie la question et la reponse a Maxence pour ameliorer Lucie.</div>
            <div class="mail-panel">
              <strong>Connexion simple</strong>
              <div class="report-note">Une seule validation connecte les mails et le calendrier. Lucie ne voit jamais le mot de passe.</div>
              <div class="mail-actions">
                <button id="connect-google" type="button">Google</button>
                <button id="connect-microsoft" type="button">Microsoft</button>
              </div>
              <div class="mail-actions">
                <button id="read-mails" type="button">Lire mes mails</button>
                <button id="read-calendar" type="button">Voir calendrier</button>
              </div>
              <div class="mail-status" id="mail-status">Connexion mail non verifiee.</div>
              <div class="mail-status" id="calendar-status">Connexion calendrier non verifiee.</div>
            </div>
          </div>
          <div class="chat-list" id="chat-list"></div>
        </aside>

        <section class="chat-card">
        <div class="chat-head">
          <div class="chat-copy">
            <strong>Comment puis-je t'aider ?</strong>
            <span>Pose une question, demande une explication, resume un texte, ou corrige une phrase.</span>
          </div>
          <div class="chips">
            <button type="button" data-fill="explique ce code">Expliquer</button>
            <button type="button" data-fill="corrige ca">Corriger</button>
            <button type="button" data-fill="resume ca">Resumer</button>
            <button type="button" data-fill="j'ai une question">Question</button>
            <button type="button" data-fill="quiz">Quiz</button>
            <button type="button" data-fill="que sais-tu de moi ?">Memoire</button>
          </div>
        </div>

        <div class="chat-log" id="chat-log"></div>

        <div class="composer">
          <form id="chat-form" autocomplete="off">
            <textarea id="message" autocomplete="off" spellcheck="false" placeholder="Message Lucie..."></textarea>
            <div class="actions">
              <div class="mini">Astuce: <code>/teach question | reponse</code> pour lui apprendre une reponse.</div>
              <button type="submit">Envoyer</button>
            </div>
          </form>
          <div class="hint" id="hint">Prete a repondre.</div>
        </div>
        </section>
      </div>
    </main>
  </div>

  <div class="onboarding" id="onboarding" hidden>
    <form class="onboarding-card" id="onboarding-form" autocomplete="off">
      <div>
        <h2>Bienvenue dans Lucie</h2>
        <p>Quelques questions rapides pour adapter les reponses et mieux t'aider des le debut.</p>
      </div>
      <div class="onboarding-grid">
        <div class="onboarding-field">
          <label for="onboard-name">Comment tu t'appelles ?</label>
          <input id="onboard-name" name="name" placeholder="Ton prenom">
        </div>
        <div class="onboarding-field">
          <label for="onboard-style">Tu preferes quel style ?</label>
          <select id="onboard-style" name="style">
            <option value="simple">Simple et clair</option>
            <option value="court">Tres court</option>
            <option value="detaille">Detaille avec exemples</option>
            <option value="prof">Comme un prof</option>
          </select>
        </div>
        <div class="onboarding-field full">
          <label for="onboard-goal">Tu veux surtout utiliser Lucie pour quoi ?</label>
          <input id="onboard-goal" name="goal" placeholder="devoirs, code, organisation, robot, idees...">
        </div>
        <div class="onboarding-field full">
          <label for="onboard-topics">Quels sujets t'interessent ?</label>
          <input id="onboard-topics" name="topics" placeholder="IA, espace, maths, histoire, programmation...">
        </div>
      </div>
      <div class="onboarding-actions">
        <button type="button" id="onboarding-skip">Passer</button>
        <button type="submit" id="onboarding-start">Commencer</button>
      </div>
    </form>
  </div>

  <script>
    const API_KEY = "__IA_API_KEY__";
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message");
    const hint = document.getElementById("hint");
    const status = document.getElementById("status");
    const chatLog = document.getElementById("chat-log");
    const onboarding = document.getElementById("onboarding");
    const onboardingForm = document.getElementById("onboarding-form");
    const onboardingSkip = document.getElementById("onboarding-skip");
    const onboardingStart = document.getElementById("onboarding-start");
    const sidebarToggle = document.getElementById("sidebar-toggle");
    const chatCard = document.querySelector(".chat-card");
    const chatSidebar = document.querySelector(".chat-sidebar");
    const sidebarClose = document.getElementById("sidebar-close");
    const newChatButton = document.getElementById("new-chat");
    const reportBadAnswerButton = document.getElementById("report-bad-answer");
    const connectGoogleButton = document.getElementById("connect-google");
    const connectMicrosoftButton = document.getElementById("connect-microsoft");
    const readMailsButton = document.getElementById("read-mails");
    const mailStatus = document.getElementById("mail-status");
    const readCalendarButton = document.getElementById("read-calendar");
    const calendarStatus = document.getElementById("calendar-status");
    const chatList = document.getElementById("chat-list");
    const brainCard = document.createElement("aside");
    brainCard.className = "brain-card";
    brainCard.innerHTML = `
      <div class="brain-orb" aria-hidden="true">
        <span class="brain-core"></span>
        <span class="brain-node node-a"></span>
        <span class="brain-node node-b"></span>
        <span class="brain-node node-c"></span>
        <span class="brain-node node-d"></span>
        <span class="brain-node node-e"></span>
        <span class="brain-node node-f"></span>
      </div>
      <div class="brain-panel">
        <div class="brain-title">
          <strong>Memoire de Lucie</strong>
          <span id="brain-mode">chargement...</span>
        </div>
        <div class="brain-focus">
          <span class="label">Sujet actif</span>
          <div class="value" id="brain-subject">Aucun pour l'instant</div>
          <div class="sub" id="brain-summary">Dès que tu parles d'un sujet, je garde le fil ici.</div>
        </div>
        <div class="brain-grid">
          <div class="brain-stat"><span class="k">Mémoire</span><span class="v" id="brain-memory">0</span></div>
          <div class="brain-stat"><span class="k">Thèmes</span><span class="v" id="brain-subjects">0</span></div>
          <div class="brain-stat"><span class="k">Documents</span><span class="v" id="brain-docs">0</span></div>
          <div class="brain-stat"><span class="k">Exemples</span><span class="v" id="brain-examples">0</span></div>
        </div>
        <div class="brain-feed" id="brain-feed">
          <div class="brain-empty">Les sujets gardés en mémoire apparaissent ici. Essaie par exemple: <strong>Parle-moi de Céline Dion</strong>, puis <strong>et son mari ?</strong></div>
        </div>
      </div>
    `;
    const brainMode = brainCard.querySelector("#brain-mode");
    const brainSubject = brainCard.querySelector("#brain-subject");
    const brainSummary = brainCard.querySelector("#brain-summary");
    const brainMemory = brainCard.querySelector("#brain-memory");
    const brainSubjects = brainCard.querySelector("#brain-subjects");
    const brainDocs = brainCard.querySelector("#brain-docs");
    const brainExamples = brainCard.querySelector("#brain-examples");
    const brainFeed = brainCard.querySelector("#brain-feed");

    function escapeText(text) {
      return String(text || "");
    }

    const STORE_KEY = "lucie_conversations_v1";
    const PROFILE_KEY = "lucie_user_profile_v1";
    let conversations = [];
    let activeConversationId = "";
    let userProfile = loadProfile();

    function loadProfile() {
      try {
        const profile = JSON.parse(localStorage.getItem(PROFILE_KEY) || "null");
        return profile && typeof profile === "object" ? profile : null;
      } catch (error) {
        return null;
      }
    }

    function saveProfile(profile) {
      userProfile = profile;
      localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    }

    async function saveProfileToServer(profile) {
      try {
        await fetch("/api/profile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ profile }),
        });
      } catch (error) {
        console.warn("Profile sync failed", error);
      }
    }

    async function loadProfileFromServer() {
      if (userProfile) return;
      try {
        const response = await fetch("/api/profile");
        const data = await response.json();
        if (data.profile && Object.keys(data.profile).length) {
          saveProfile(data.profile);
        }
      } catch (error) {
        console.warn("Profile load failed", error);
      }
    }

    function profileName() {
      return userProfile && userProfile.name ? String(userProfile.name).trim() : "";
    }

    function profileWelcomeText() {
      if (!userProfile) {
        return "Lucie est prete. Envoie un message pour commencer.";
      }
      const name = profileName();
      const hello = name ? `Salut ${name}, je suis prete.` : "Je suis prete.";
      const goal = userProfile.goal ? ` Je vais t'aider surtout pour: ${userProfile.goal}.` : "";
      const style = userProfile.style ? ` Style choisi: ${userProfile.style}.` : "";
      return `${hello}${goal}${style}`;
    }

    function loadConversations() {
      try {
        conversations = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
      } catch (error) {
        conversations = [];
      }
      if (!Array.isArray(conversations)) {
        conversations = [];
      }
      if (!conversations.length) {
        createConversation(false);
      } else if (!activeConversationId) {
        activeConversationId = conversations[0].id;
      }
    }

    function saveConversations() {
      localStorage.setItem(STORE_KEY, JSON.stringify(conversations.slice(0, 30)));
    }

    async function loadServerConversations() {
      try {
        const response = await fetch("/api/conversations");
        const data = await response.json();
        if (Array.isArray(data.conversations) && data.conversations.length) {
          conversations = data.conversations;
          activeConversationId = conversations[0].id;
          saveConversations();
          renderConversation();
          renderChatList();
        }
      } catch (error) {
        console.warn("Server history unavailable", error);
      }
    }

    function currentConversation() {
      return conversations.find((item) => item.id === activeConversationId) || conversations[0];
    }

    function titleFrom(text) {
      const value = shortText(text || "Nouveau chat", 34);
      return value || "Nouveau chat";
    }

    function createConversation(render = true) {
      const conversation = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        title: "Nouveau chat",
        messages: [],
        updatedAt: Date.now(),
      };
      conversations.unshift(conversation);
      activeConversationId = conversation.id;
      saveConversations();
      if (render) {
        renderConversation();
      }
      renderChatList();
    }

    function rememberMessage(role, text) {
      const conversation = currentConversation();
      if (!conversation) return;
      conversation.messages.push({ role, text });
      conversation.updatedAt = Date.now();
      if (role === "user" && conversation.title === "Nouveau chat") {
        conversation.title = titleFrom(text);
      }
      conversations = [conversation, ...conversations.filter((item) => item.id !== conversation.id)];
      activeConversationId = conversation.id;
      saveConversations();
      renderChatList();
    }

    function renderChatList() {
      if (!chatList) return;
      chatList.innerHTML = "";
      conversations.forEach((conversation) => {
        const button = document.createElement("button");
        button.className = `chat-item${conversation.id === activeConversationId ? " active" : ""}`;
        button.type = "button";
        button.textContent = conversation.title || "Nouveau chat";
        button.addEventListener("click", () => {
          activeConversationId = conversation.id;
          renderConversation();
          renderChatList();
        });
        chatList.appendChild(button);
      });
    }

    function renderConversation() {
      chatLog.innerHTML = "";
      const conversation = currentConversation();
      const messages = conversation ? conversation.messages : [];
      if (!messages.length) {
        addBubble("system", profileWelcomeText(), false);
        return;
      }
      messages.forEach((message) => addBubble(message.role, message.text, false));
    }

    function latestProblemReport() {
      const conversation = currentConversation();
      const messages = conversation ? conversation.messages : [];
      const lastUser = [...messages].reverse().find((message) => message.role === "user");
      const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
      if (!lastUser) {
        return null;
      }
      return {
        title: conversation.title || "Discussion Lucie",
        question: String(lastUser.text || "").trim(),
        answer: String(lastAssistant ? lastAssistant.text : "").trim(),
      };
    }

    async function reportBadAnswer(reportOverride = null) {
      const report = reportOverride || latestProblemReport();
      if (!report || !report.question) {
        alert("Il faut d'abord poser une question a Lucie.");
        return;
      }
      const correction = prompt("Explique le probleme ou ecris la bonne reponse pour Maxence :", "");
      const payload = {
        ...report,
        correction: correction || "",
        profile: userProfile || {},
        url: window.location.href,
      };
      try {
        const response = await fetch("/api/report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (response.ok) {
          setStatus("Signalement envoye a l'admin", true);
          alert("C'est envoye. Maxence le verra dans la page admin.");
          return;
        }
      } catch (error) {
        console.warn("Report failed, opening mail fallback", error);
      }
      const body = [
        "Bonjour Maxence,",
        "",
        "Lucie a eu un probleme sur cette question :",
        "",
        `Question : ${report.question}`,
        "",
        "Reponse de Lucie :",
        report.answer || "(pas de reponse trouvee)",
        "",
        "Ce qu'il faudrait corriger :",
        correction || "",
        "Envoye depuis l'app Lucie.",
      ].join(String.fromCharCode(10));
      const subject = `Probleme Lucie - ${report.title}`.slice(0, 120);
      const mailto = `mailto:kleibermaxence@gmail.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body.slice(0, 6000))}`;
      const link = document.createElement("a");
      link.href = mailto;
      link.target = "_blank";
      link.rel = "noopener";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setStatus("Signalement prepare");
    }

    function confidenceText(confidence) {
      const value = Number(confidence || 0);
      if (value >= 0.78) return "Confiance haute";
      if (value >= 0.52) return "Confiance moyenne";
      return "A verifier";
    }

    function shortAnswer(text, limit = 520) {
      const value = String(text || "").trim();
      if (value.length <= limit) return value;
      const cut = value.slice(0, limit);
      const end = Math.max(cut.lastIndexOf("."), cut.lastIndexOf("\n"), cut.lastIndexOf(" "));
      return `${cut.slice(0, end > 220 ? end + 1 : limit).trim()}\n\n...`;
    }

    function addBubble(role, text, save = true, meta = {}) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;
      const fullText = String(text || "");
      const collapsedText = role === "assistant" && meta.collapsible !== false ? shortAnswer(fullText) : fullText;
      const textNode = document.createElement("div");
      textNode.className = "bubble-text";
      textNode.textContent = escapeText(collapsedText);
      bubble.appendChild(textNode);
      if (role === "assistant" && meta.actions !== false) {
        const actions = document.createElement("div");
        actions.className = "bubble-actions";
        if (typeof meta.confidence === "number") {
          const pill = document.createElement("span");
          pill.className = "confidence-pill";
          pill.textContent = confidenceText(meta.confidence);
          actions.appendChild(pill);
        }
        const improve = document.createElement("button");
        improve.type = "button";
        improve.textContent = "Ameliorer cette reponse";
        improve.addEventListener("click", () => {
          const conversation = currentConversation();
          const messages = conversation ? conversation.messages : [];
          const lastUser = [...messages].reverse().find((message) => message.role === "user");
          reportBadAnswer({
            title: conversation ? conversation.title : "Discussion Lucie",
            question: String(lastUser ? lastUser.text : "").trim(),
            answer: String(text || "").trim(),
          });
        });
        actions.appendChild(improve);
        if (collapsedText !== fullText) {
          const expand = document.createElement("button");
          expand.type = "button";
          expand.textContent = "Developper";
          expand.addEventListener("click", () => {
            const expanded = expand.dataset.expanded === "1";
            textNode.textContent = expanded ? escapeText(collapsedText) : escapeText(fullText);
            expand.dataset.expanded = expanded ? "0" : "1";
            expand.textContent = expanded ? "Developper" : "Reduire";
            chatLog.scrollTop = chatLog.scrollHeight;
          });
          actions.appendChild(expand);
        }
        bubble.appendChild(actions);
      }
      chatLog.appendChild(bubble);
      chatLog.scrollTop = chatLog.scrollHeight;
      if (save && role !== "system") {
        rememberMessage(role, text);
      }
      return bubble;
    }

    function ensureWelcome() {
      if (!chatLog.children.length) {
        addBubble("system", profileWelcomeText());
      }
    }

    function authHeaders(extra = {}) {
      return {
        ...extra,
        "Authorization": `Bearer ${API_KEY}`,
      };
    }

    function setStatus(text, accent = false) {
      status.innerHTML = `<span class="dot"></span><span>${text}</span>`;
      status.style.borderColor = accent ? "#10a37f" : "#e5e7eb";
    }

    function setMailStatus(text) {
      if (mailStatus) {
        mailStatus.textContent = text;
      }
    }

    function setCalendarStatus(text) {
      if (calendarStatus) {
        calendarStatus.textContent = text;
      }
    }

    function shortText(text, limit = 120) {
      const value = String(text || "").replace(/\s+/g, " ").trim();
      if (!value) return "";
      if (value.length <= limit) return value;
      const cut = value.slice(0, limit);
      const idx = cut.lastIndexOf(" ");
      return `${(idx > 24 ? cut.slice(0, idx) : cut).trim()}...`;
    }

    function renderBrain(data = {}) {
      const mode = data.mode === "openai" ? `IA active · ${data.model}` : `Local · ${data.model}`;
      brainMode.textContent = mode;
      brainSubject.textContent = shortText(data.last_subject || "Aucun pour l'instant", 34) || "Aucun pour l'instant";
      brainSummary.textContent = shortText(data.conversation_summary || "Dès que tu parles d'un sujet, je garde le fil ici.", 210) || "Dès que tu parles d'un sujet, je garde le fil ici.";
      brainMemory.textContent = String(data.memory_count ?? 0);
      brainSubjects.textContent = String(data.subjects_count ?? 0);
      brainDocs.textContent = String(data.document_count ?? 0);
      brainExamples.textContent = String(data.example_count ?? 0);

      const briefs = data.subject_briefs || {};
      const entries = Object.entries(briefs).slice(-4).reverse();
      brainFeed.innerHTML = "";
      if (!entries.length) {
        const empty = document.createElement("div");
        empty.className = "brain-empty";
        empty.innerHTML = "Les sujets gardés en mémoire apparaissent ici. Essaie par exemple: <strong>Parle-moi de Céline Dion</strong>, puis <strong>et son mari ?</strong>";
        brainFeed.appendChild(empty);
      } else {
        entries.forEach(([subject, brief]) => {
          const card = document.createElement("div");
          card.className = "brain-chip";
          card.innerHTML = `<strong>${escapeText(subject)}</strong>${escapeText(shortText(brief, 180))}`;
          brainFeed.appendChild(card);
        });
      }
    }

    async function loadStatus() {
      try {
        const res = await fetch("/api/status", {
          headers: authHeaders(),
        });
        const data = await res.json();
        if (data.mode === "openai") {
          setStatus(`Mode IA active: ${data.model}`);
        } else {
          setStatus(`Mode local: ${data.model}`);
        }
        renderBrain(data);
        if (data.warning) {
          hint.textContent = data.warning;
        }
      } catch (error) {
        setStatus("Statut indisponible");
      }
    }

    async function loadEmailStatus() {
      try {
        const res = await fetch("/api/email/status", { headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Statut mail indisponible");
        }
        const providers = data.providers || {};
        const connected = Object.entries(providers)
          .filter(([, info]) => info.connected)
          .map(([name]) => name);
        if (connected.length) {
          setMailStatus(`Connecte: ${connected.join(", ")}.`);
          return;
        }
        const configured = Object.values(providers).some((info) => info.configured);
        setMailStatus(configured ? "Pret: connecte Gmail ou Outlook." : "Admin: ajoute les cles OAuth sur Render.");
      } catch (error) {
        setMailStatus("Impossible de verifier les mails.");
      }
    }

    function connectEmail(provider) {
      window.location.href = `/connect/${provider}`;
    }

    function connectAccount(provider) {
      window.location.href = `/connect/account/${provider}`;
    }

    function showOnboardingIfNeeded() {
      if (!userProfile && onboarding) {
        onboarding.hidden = false;
        const nameInput = document.getElementById("onboard-name");
        if (nameInput) {
          setTimeout(() => nameInput.focus(), 50);
        }
      }
    }

    function finishOnboarding(profile) {
      saveProfile(profile);
      saveProfileToServer(profile);
      if (onboarding) {
        onboarding.hidden = true;
      }
      const name = profile.name ? ` ${profile.name}` : "";
      const topics = profile.topics ? ` Je retiens aussi tes sujets: ${profile.topics}.` : "";
      addBubble("assistant", `Parfait${name}. Je vais adapter mes reponses pour toi.${topics}`);
      hint.textContent = "Profil enregistre dans ce navigateur.";
    }

    function collectOnboardingProfile() {
      return {
        name: String(document.getElementById("onboard-name")?.value || "").trim(),
        style: String(document.getElementById("onboard-style")?.value || "simple").trim(),
        goal: String(document.getElementById("onboard-goal")?.value || "").trim(),
        topics: String(document.getElementById("onboard-topics")?.value || "").trim(),
        createdAt: new Date().toISOString(),
      };
    }

    async function readConnectedMails() {
      setMailStatus("Lecture des derniers mails...");
      try {
        const res = await fetch("/api/email/inbox", { headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Mails indisponibles");
        }
        const mails = data.messages || [];
        if (!mails.length) {
          addBubble("assistant", "Je n'ai trouve aucun mail recent connecte pour l'instant.");
          setMailStatus("Aucun mail recent.");
          return;
        }
        const summary = mails.map((mail, index) => {
          const from = mail.from || "Expediteur inconnu";
          const subject = mail.subject || "Sans objet";
          const preview = mail.preview ? ` - ${mail.preview}` : "";
          return `${index + 1}. ${from}: ${subject}${preview}`;
        }).join(String.fromCharCode(10));
        addBubble("assistant", `Voici les derniers mails que je peux lire:\n${summary}`);
        setMailStatus(`${mails.length} mail(s) lus.`);
      } catch (error) {
        const text = String(error.message || error);
        addBubble("assistant", text);
        setMailStatus(text);
      }
    }

    async function loadCalendarStatus() {
      try {
        const res = await fetch("/api/calendar/status", { headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Statut calendrier indisponible");
        }
        const providers = data.providers || {};
        const connected = Object.entries(providers)
          .filter(([, info]) => info.connected)
          .map(([name]) => name.replace("_", " "));
        if (connected.length) {
          setCalendarStatus(`Connecte: ${connected.join(", ")}.`);
          return;
        }
        const configured = Object.values(providers).some((info) => info.configured);
        setCalendarStatus(configured ? "Pret: connecte Google ou Outlook." : "Admin: ajoute les cles OAuth calendrier.");
      } catch (error) {
        setCalendarStatus("Impossible de verifier le calendrier.");
      }
    }

    function connectCalendar(provider) {
      window.location.href = `/connect/calendar/${provider}`;
    }

    async function readConnectedCalendar() {
      setCalendarStatus("Lecture du calendrier...");
      try {
        const res = await fetch("/api/calendar/events", { headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Calendrier indisponible");
        }
        const events = data.events || [];
        if (!events.length) {
          addBubble("assistant", "Je n'ai trouve aucun evenement dans les prochains jours.");
          setCalendarStatus("Aucun evenement prochain.");
          return;
        }
        const summary = events.map((event, index) => {
          const when = event.start || "date inconnue";
          const title = event.title || "Sans titre";
          const place = event.location ? ` - ${event.location}` : "";
          return `${index + 1}. ${when}: ${title}${place}`;
        }).join(String.fromCharCode(10));
        addBubble("assistant", `Voici tes prochains evenements:${String.fromCharCode(10)}${summary}`);
        setCalendarStatus(`${events.length} evenement(s) lus.`);
      } catch (error) {
        const text = String(error.message || error);
        addBubble("assistant", text);
        setCalendarStatus(text);
      }
    }

    if (sidebarClose) {
      sidebarClose.addEventListener("click", () => {
        document.body.classList.add("sidebar-closed");
      });
    }

    if (newChatButton) {
      newChatButton.addEventListener("click", () => {
        createConversation(true);
        input.focus();
      });
    }

    if (reportBadAnswerButton) {
      reportBadAnswerButton.addEventListener("click", reportBadAnswer);
    }

    if (connectGoogleButton) {
      connectGoogleButton.addEventListener("click", () => connectAccount("google"));
    }

    if (connectMicrosoftButton) {
      connectMicrosoftButton.addEventListener("click", () => connectAccount("microsoft"));
    }

    if (readMailsButton) {
      readMailsButton.addEventListener("click", readConnectedMails);
    }

    if (readCalendarButton) {
      readCalendarButton.addEventListener("click", readConnectedCalendar);
    }

    if (onboardingForm) {
      onboardingForm.addEventListener("submit", (event) => {
        event.preventDefault();
        finishOnboarding(collectOnboardingProfile());
      });
    }

    if (onboardingStart) {
      onboardingStart.addEventListener("click", (event) => {
        event.preventDefault();
        finishOnboarding(collectOnboardingProfile());
      });
    }

    if (onboardingSkip) {
      onboardingSkip.addEventListener("click", () => {
        finishOnboarding({
          name: "",
          style: "simple",
          goal: "",
          topics: "",
          skipped: true,
          createdAt: new Date().toISOString(),
        });
      });
    }

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) {
        return;
      }

      ensureWelcome();
      addBubble("user", message);
      const activeConversation = currentConversation();
      hint.textContent = "Analyse en cours...";
      setStatus("Analyse", true);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({
            message,
            profile: userProfile || {},
            conversation_id: activeConversation ? activeConversation.id : "",
            conversation_title: activeConversation ? activeConversation.title : "",
          }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Requete refusee");
        }

        addBubble("assistant", data.answer || "OK", true, { confidence: data.confidence });

        const parts = [];
        if (data.note) parts.push(data.note);
        if (data.intent && data.intent.label) {
          parts.push(`Intention: ${data.intent.label} (${Math.round(data.intent.confidence * 100)}%)`);
        }
        if (typeof data.memory_count === "number") {
          parts.push(`Memoire: ${data.memory_count}`);
        }
        if (typeof data.subjects_count === "number") {
          parts.push(`Themes: ${data.subjects_count}`);
        }
        if (typeof data.document_count === "number") {
          parts.push(`Documents: ${data.document_count}`);
        }
        if (typeof data.example_count === "number") {
          parts.push(`Exemples: ${data.example_count}`);
        }
        if (typeof data.memory_sources_count === "number") {
          parts.push(`Sources memoire: ${data.memory_sources_count}`);
        }
        if (data.confidence_label) {
          parts.push(`Confiance: ${data.confidence_label}`);
        }
        if (data.pending_action) {
          parts.push(`Mode: ${data.pending_action}`);
        }
        hint.textContent = parts.join(" | ") || "Pret.";
        renderBrain(data);
        setStatus("Réponse", false);
      } catch (error) {
        const text = String(error.message || error);
        addBubble("assistant", text);
        hint.textContent = "Le serveur n'a pas repondu correctement.";
        setStatus("Erreur", false);
      } finally {
        input.value = "";
        input.focus();
      }
    });

    loadConversations();
    renderConversation();
    renderChatList();
    loadProfileFromServer().then(() => showOnboardingIfNeeded());
    loadServerConversations();
    input.focus();
    loadStatus();
    loadEmailStatus();
    loadCalendarStatus();
  </script>
</body>
</html>
"""


ADMIN_PAGE = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Admin Lucie</title>
  <style>
    :root { color-scheme: light; --bg: #f7f7f8; --panel: #ffffff; --text: #111827; --muted: #6b7280; --border: #e5e7eb; --accent: #10a37f; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: var(--bg); color: var(--text); }
    header { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 28px; background: rgba(255,255,255,.92); border-bottom: 1px solid var(--border); backdrop-filter: blur(10px); }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 22px; }
    h2 { font-size: 18px; margin-bottom: 12px; }
    main { width: min(1180px, calc(100% - 32px)); margin: 24px auto 48px; display: grid; gap: 18px; }
    body.admin-locked main { display: none; }
    .toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .toolbar input { width: min(420px, 100%); border: 1px solid var(--border); border-radius: 12px; padding: 11px 12px; font: inherit; }
    button, a.button { border: 1px solid var(--border); border-radius: 12px; background: #fff; color: var(--text); padding: 10px 14px; font: inherit; font-weight: 700; cursor: pointer; text-decoration: none; }
    button.primary { background: #111827; color: #fff; border-color: #111827; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 18px; box-shadow: 0 8px 28px rgba(15, 23, 42, .05); }
    .stat span { display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .stat strong { font-size: 26px; }
    .columns { display: grid; grid-template-columns: 1.1fr .9fr; gap: 18px; align-items: start; }
    .list { display: grid; gap: 10px; max-height: 520px; overflow: auto; padding-right: 4px; }
    .item { border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: #fff; }
    .item strong { display: block; margin-bottom: 6px; }
    .muted { color: var(--muted); font-size: 13px; }
    textarea, input[type="text"] { width: 100%; border: 1px solid var(--border); border-radius: 12px; padding: 12px; font: inherit; resize: vertical; }
    form { display: grid; gap: 10px; }
    .summary { white-space: pre-wrap; line-height: 1.5; color: #374151; }
    .lock-screen { width: min(520px, calc(100% - 32px)); margin: 80px auto; display: grid; gap: 16px; text-align: center; }
    .lock-card { background: #fff; border: 1px solid var(--border); border-radius: 18px; padding: 28px; box-shadow: 0 18px 60px rgba(15, 23, 42, .08); display: grid; gap: 14px; }
    .lock-card input { width: 100%; border: 1px solid var(--border); border-radius: 14px; padding: 14px; font: inherit; text-align: center; font-size: 20px; letter-spacing: 4px; }
    body:not(.admin-locked) .lock-screen { display: none; }
    @media (max-width: 840px) { .grid, .columns { grid-template-columns: 1fr; } header { align-items: flex-start; flex-direction: column; } }
  </style>
</head>
<body class="admin-locked">
  <header>
    <div>
      <h1>Admin Lucie</h1>
      <p class="muted">Memoire, signalements et apprentissage rapide.</p>
    </div>
    <div class="toolbar">
      <input id="api-key" type="password" placeholder="Code admin">
      <button id="save-key" type="button">Entrer</button>
      <button id="refresh" class="primary" type="button">Actualiser</button>
      <a class="button" href="/">Retour au chat</a>
    </div>
  </header>
  <section class="lock-screen" id="lock-screen">
    <div class="lock-card">
      <h2>Code admin</h2>
      <p class="muted">Entre le code pour acceder a la page Maxence.</p>
      <input id="lock-code" type="password" inputmode="numeric" placeholder="Code">
      <button id="unlock" class="primary" type="button">Acceder a l'admin</button>
      <div id="lock-status" class="muted"></div>
    </div>
  </section>
  <main id="admin-content" aria-hidden="true">
    <section class="grid">
      <div class="card stat"><span>Souvenirs</span><strong id="memory-count">0</strong></div>
      <div class="card stat"><span>Sujets</span><strong id="subjects-count">0</strong></div>
      <div class="card stat"><span>Documents</span><strong id="document-count">0</strong></div>
      <div class="card stat"><span>Signalements</span><strong id="reports-count">0</strong></div>
      <div class="card stat"><span>Utilisateurs</span><strong id="users-count">0</strong></div>
      <div class="card stat"><span>Chats serveur</span><strong id="server-chats-count">0</strong></div>
    </section>
    <section class="columns">
      <div class="card">
        <h2>Memoire de discussion</h2>
        <div id="summary" class="summary muted">Chargement...</div>
        <button id="compact-memory" type="button" style="margin-top:12px">Compacter la memoire</button>
      </div>
      <div class="card">
        <h2>Apprendre une correction</h2>
        <form id="teach-form">
          <input id="teach-question" type="text" placeholder="Question exacte ou proche">
          <textarea id="teach-answer" rows="5" placeholder="Bonne reponse de Lucie"></textarea>
          <button class="primary" type="submit">Apprendre</button>
          <div id="teach-status" class="muted"></div>
        </form>
      </div>
    </section>
    <section class="columns">
      <div class="card">
        <h2>Base de connaissances</h2>
        <form id="knowledge-form">
          <input id="knowledge-category" type="text" placeholder="Categorie: ecole, code, robotique...">
          <input id="knowledge-title" type="text" placeholder="Titre de la fiche">
          <textarea id="knowledge-content" rows="6" placeholder="Information fiable que Lucie doit utiliser"></textarea>
          <button class="primary" type="submit">Ajouter la fiche</button>
          <div id="knowledge-status" class="muted"></div>
        </form>
        <form id="search-form" style="margin-top:14px">
          <input id="search-query" type="text" placeholder="Chercher dans la base">
          <button type="submit">Rechercher</button>
          <div id="search-results" class="list"></div>
        </form>
      </div>
      <div class="card">
        <h2>Mode d'emploi rapide</h2>
        <p class="summary muted">1. Un utilisateur signale une mauvaise reponse.
2. Tu cliques sur "Corriger" dans les signalements.
3. Tu valides la bonne reponse.
4. Lucie reutilise cette correction dans les prochains chats.</p>
      </div>
    </section>
    <section class="columns">
      <div class="card">
        <h2>Dernieres questions</h2>
        <div id="recent-questions" class="list"></div>
      </div>
      <div class="card">
        <h2>Confiance faible</h2>
        <div id="low-confidence" class="list"></div>
      </div>
    </section>
    <section class="columns">
      <div class="card">
        <h2>Souvenirs</h2>
        <div id="memory-notes" class="list"></div>
      </div>
      <div class="card">
        <h2>Actions qualite</h2>
        <p class="summary muted">Avant un deploy, lance le test qualite local. Il verifie les questions importantes, les calculs, la memoire et les reponses trop vagues.</p>
      </div>
    </section>
    <section class="columns">
      <div class="card">
        <h2>Signalements des utilisateurs</h2>
        <div id="reports" class="list"></div>
      </div>
      <div class="card">
        <h2>Sujets suivis</h2>
        <div id="subjects" class="list"></div>
      </div>
    </section>
  </main>
  <script>
    const keyInput = document.getElementById("api-key");
    const lockCodeInput = document.getElementById("lock-code");
    const lockStatus = document.getElementById("lock-status");
    const saved = localStorage.getItem("lucie_admin_code");
    if (saved) {
      keyInput.value = saved;
      lockCodeInput.value = saved;
    }

    function headers(extra = {}) {
      const code = keyInput.value.trim();
      return { ...extra, "X-Admin-Code": code };
    }
    function text(value) {
      return String(value || "").replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
    }
    function set(id, value) {
      document.getElementById(id).textContent = value;
    }
    async function loadAdmin() {
      const code = keyInput.value.trim();
      if (!code) {
        document.body.classList.add("admin-locked");
        document.getElementById("admin-content").setAttribute("aria-hidden", "true");
        lockStatus.textContent = "Code obligatoire.";
        return;
      }
      const response = await fetch("/api/admin/overview", { headers: headers() });
      if (!response.ok) {
        document.body.classList.add("admin-locked");
        document.getElementById("admin-content").setAttribute("aria-hidden", "true");
        lockStatus.textContent = "Code incorrect.";
        return;
      }
      document.body.classList.remove("admin-locked");
      document.getElementById("admin-content").setAttribute("aria-hidden", "false");
      lockStatus.textContent = "";
      const data = await response.json();
      const status = data.status || {};
      set("memory-count", status.memory_count || 0);
      set("subjects-count", status.subjects_count || 0);
      set("document-count", status.document_count || 0);
      set("reports-count", (data.reports || []).length);
      set("users-count", data.user_count || 0);
      set("server-chats-count", data.server_chat_count || 0);
      document.getElementById("summary").textContent = status.conversation_summary || "Pas encore de memoire.";
      renderReports(data.reports || []);
      renderSubjects(data.subject_briefs || {});
      renderQuestionList("recent-questions", data.recent_questions || [], "Aucune question recente.");
      renderQuestionList("low-confidence", data.low_confidence || [], "Aucune reponse faible.");
      renderMemoryNotes(data.memory_notes || []);
    }
    function renderMemoryNotes(notes) {
      const box = document.getElementById("memory-notes");
      if (!notes.length) {
        box.innerHTML = '<div class="muted">Aucun souvenir direct.</div>';
        return;
      }
      box.innerHTML = notes.slice().reverse().map((note, index) => `
        <div class="item">
          <p>${text(note)}</p>
          <button type="button" data-memory-index="${index}">Supprimer</button>
        </div>
      `).join("");
      box.querySelectorAll("button[data-memory-index]").forEach((button) => {
        button.addEventListener("click", async () => {
          const note = notes.slice().reverse()[Number(button.dataset.memoryIndex || 0)] || "";
          if (!note || !confirm("Supprimer ce souvenir ?")) return;
          await fetch("/api/admin/delete-memory-note", {
            method: "POST",
            headers: headers({ "Content-Type": "application/json" }),
            body: JSON.stringify({ note }),
          });
          loadAdmin();
        });
      });
    }
    function renderQuestionList(id, questions, emptyText) {
      const box = document.getElementById(id);
      if (!questions.length) {
        box.innerHTML = `<div class="muted">${emptyText}</div>`;
        return;
      }
      box.innerHTML = questions.map((item, index) => `
        <div class="item">
          <strong>${text(item.question)}</strong>
          <div class="muted">${text(item.conversation)} - ${text(item.confidence_label || "non note")}</div>
          <p>${text(item.answer || "(pas de reponse)")}</p>
          <button type="button" data-question-list="${id}" data-question-index="${index}">Corriger</button>
        </div>
      `).join("");
      box.querySelectorAll("button[data-question-index]").forEach((button) => {
        button.addEventListener("click", () => {
          const item = questions[Number(button.dataset.questionIndex || 0)] || {};
          document.getElementById("teach-question").value = item.question || "";
          document.getElementById("teach-answer").value = item.answer || "";
          document.getElementById("teach-question").focus();
        });
      });
    }
    function renderReports(reports) {
      const box = document.getElementById("reports");
      if (!reports.length) {
        box.innerHTML = '<div class="muted">Aucun signalement pour le moment.</div>';
        return;
      }
      box.innerHTML = reports.map((item, index) => `
        <div class="item">
          <strong>${text(item.question)}</strong>
          <div class="muted">${text(item.created_at)} - ${text(item.title)}</div>
          <p>Reponse: ${text(item.answer || "(vide)")}</p>
          <p>Correction demandee: ${text(item.correction || "(non precisee)")}</p>
          <button type="button" data-report-index="${index}">Corriger</button>
        </div>
      `).join("");
      box.querySelectorAll("button[data-report-index]").forEach((button) => {
        button.addEventListener("click", () => {
          const item = reports[Number(button.dataset.reportIndex || 0)] || {};
          document.getElementById("teach-question").value = item.question || "";
          document.getElementById("teach-answer").value = item.correction || item.answer || "";
          document.getElementById("teach-question").focus();
        });
      });
    }
    function renderSubjects(subjects) {
      const entries = Object.entries(subjects);
      const box = document.getElementById("subjects");
      if (!entries.length) {
        box.innerHTML = '<div class="muted">Aucun sujet suivi.</div>';
        return;
      }
      box.innerHTML = entries.slice(-40).reverse().map(([subject, brief]) => `
        <div class="item"><strong>${text(subject)}</strong><p>${text(brief)}</p></div>
      `).join("");
    }
    document.getElementById("save-key").addEventListener("click", () => {
      localStorage.setItem("lucie_admin_code", keyInput.value.trim());
      lockCodeInput.value = keyInput.value.trim();
      loadAdmin();
    });
    document.getElementById("unlock").addEventListener("click", () => {
      keyInput.value = lockCodeInput.value.trim();
      localStorage.setItem("lucie_admin_code", keyInput.value);
      loadAdmin();
    });
    lockCodeInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        keyInput.value = lockCodeInput.value.trim();
        localStorage.setItem("lucie_admin_code", keyInput.value);
        loadAdmin();
      }
    });
    document.getElementById("refresh").addEventListener("click", loadAdmin);
    document.getElementById("compact-memory").addEventListener("click", async () => {
      const response = await fetch("/api/admin/compact-memory", {
        method: "POST",
        headers: headers({ "Content-Type": "application/json" }),
        body: "{}",
      });
      alert(response.ok ? "Memoire compactee." : "Impossible de compacter.");
      loadAdmin();
    });
    document.getElementById("teach-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = document.getElementById("teach-question").value.trim();
      const answer = document.getElementById("teach-answer").value.trim();
      const status = document.getElementById("teach-status");
      if (!question || !answer) {
        status.textContent = "Question et reponse obligatoires.";
        return;
      }
      const response = await fetch("/api/teach", {
        method: "POST",
        headers: headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ question, answer }),
      });
      status.textContent = response.ok ? "Lucie a appris cette correction." : "Impossible d'apprendre pour le moment.";
      if (response.ok) {
        document.getElementById("teach-form").reset();
        loadAdmin();
      }
    });
    document.getElementById("knowledge-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const category = document.getElementById("knowledge-category").value.trim() || "general";
      const title = document.getElementById("knowledge-title").value.trim() || "Fiche";
      const content = document.getElementById("knowledge-content").value.trim();
      const status = document.getElementById("knowledge-status");
      if (!content) {
        status.textContent = "Contenu obligatoire.";
        return;
      }
      const response = await fetch("/api/document", {
        method: "POST",
        headers: headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ title: `[${category}] ${title}`, content }),
      });
      status.textContent = response.ok ? "Fiche ajoutee a la base de connaissances." : "Impossible d'ajouter la fiche.";
      if (response.ok) {
        document.getElementById("knowledge-form").reset();
        loadAdmin();
      }
    });
    document.getElementById("search-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = document.getElementById("search-query").value.trim();
      const box = document.getElementById("search-results");
      if (!query) {
        box.innerHTML = '<div class="muted">Entre une recherche.</div>';
        return;
      }
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      const results = data.results || [];
      box.innerHTML = results.length
        ? results.map((item) => `<div class="item"><strong>${text(item.source || "Document")}</strong><p>${text(item.content)}</p><div class="muted">Score ${Math.round((item.score || 0) * 100)}%</div></div>`).join("")
        : '<div class="muted">Aucun resultat trouve.</div>';
    });
    loadAdmin();
  </script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    bot: LearningBot
    api_key: str = ""
    admin_code: str = "042724"
    reports_path: Path = Path(__file__).with_name("reports.json")
    users_path: Path = Path(__file__).with_name("users.json")
    chats_path: Path = Path(__file__).with_name("server_chats.json")
    last_memory_save_at: float = 0.0
    pending_memory_saves: int = 0
    email_states: dict[str, str] = {}
    email_tokens: dict[str, dict[str, object]] = {}
    calendar_states: dict[str, str] = {}
    calendar_tokens: dict[str, dict[str, object]] = {}
    account_states: dict[str, str] = {}

    def log_message(self, format: str, *args: object) -> None:
        return

    def _robot_status_payload(self) -> dict[str, object]:
        enabled = os.getenv("ENABLE_ROBOT_STATUS", "").strip().lower()
        if enabled in {"1", "true", "yes", "on"}:
            return self.bot.robot_bridge_status()
        return {
            "ok": False,
            "message": "Robot status disabled.",
            "connected": False,
            "serial_port": "",
            "board_type": "",
            "last_action": "disabled",
            "last_payload": {},
            "last_at": 0.0,
            "last_error": "",
            "command_count": 0,
            "uptime_seconds": 0.0,
        }

    def _status_payload(self) -> dict[str, object]:
        return {
            "ok": True,
            "warning": self.bot.startup_warning,
            "model": self.bot.model,
            "mode": "openai" if self.bot.api_available else "local",
            "knowledge_source": "dify" if self.bot.dify_client.is_ready() else "none",
            "pending_action": self.bot.pending_action,
            "memory_count": len(self.bot.memory_notes),
            "memory_sources_count": len(self.bot.memory_sources),
            "preferences_count": len(self.bot.preferences),
            "subjects_count": len(self.bot.subject_memory),
            "document_count": len(self.bot.documents),
            "example_count": self.bot.total_example_count(),
            "last_subject": self.bot.last_subject,
            "conversation_summary": self.bot.get_conversation_summary(),
            "subject_briefs": self.bot.list_subject_briefs(),
            "robot_status": self._robot_status_payload(),
        }

    def _load_reports(self) -> list[dict[str, object]]:
        if not self.reports_path.exists():
            return []
        try:
            raw = json.loads(self.reports_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return raw if isinstance(raw, list) else []

    def _save_report(self, report: dict[str, object]) -> list[dict[str, object]]:
        reports = self._load_reports()
        reports.append(report)
        reports = reports[-300:]
        self.reports_path.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return reports

    def _load_json_dict(self, path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_json_dict(self, path: Path, payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _current_profile(self) -> dict[str, object]:
        users = self._load_json_dict(self.users_path)
        profile = users.get(self._session_id(), {})
        return profile if isinstance(profile, dict) else {}

    def _save_profile(self, profile: dict[str, object]) -> dict[str, object]:
        allowed = {"name", "style", "goal", "topics", "createdAt", "updatedAt", "skipped"}
        clean = {key: str(value).strip()[:500] for key, value in profile.items() if key in allowed and value is not None}
        clean["updatedAt"] = datetime.now(timezone.utc).isoformat()
        users = self._load_json_dict(self.users_path)
        users[self._session_id()] = clean
        self._write_json_dict(self.users_path, users)
        return clean

    def _load_session_chats(self) -> list[dict[str, object]]:
        chats = self._load_json_dict(self.chats_path)
        items = chats.get(self._session_id(), [])
        return items if isinstance(items, list) else []

    def _save_chat_turn(
        self,
        conversation_id: str,
        title: str,
        question: str,
        answer: str,
        confidence: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        conversation_id = conversation_id[:96] or secrets.token_urlsafe(12)
        title = (title or "Nouveau chat").strip()[:80] or "Nouveau chat"
        chats = self._load_json_dict(self.chats_path)
        session_id = self._session_id()
        conversations = chats.get(session_id, [])
        if not isinstance(conversations, list):
            conversations = []
        conversation = next((item for item in conversations if isinstance(item, dict) and item.get("id") == conversation_id), None)
        if conversation is None:
            conversation = {
                "id": conversation_id,
                "title": title,
                "messages": [],
                "createdAt": int(time.time() * 1000),
            }
            conversations.insert(0, conversation)
        conversation["title"] = title
        conversation["updatedAt"] = int(time.time() * 1000)
        messages = conversation.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        messages.extend([
            {"role": "user", "text": question},
            {
                "role": "assistant",
                "text": answer,
                "confidence": (confidence or {}).get("confidence"),
                "confidence_label": (confidence or {}).get("confidence_label"),
            },
        ])
        conversation["messages"] = messages[-120:]
        chats[session_id] = conversations[:30]
        self._write_json_dict(self.chats_path, chats)
        return chats[session_id]

    def _recent_server_questions(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        chats = self._load_json_dict(self.chats_path)
        for session_id, conversations in chats.items():
            if not isinstance(conversations, list):
                continue
            for conversation in conversations:
                if not isinstance(conversation, dict):
                    continue
                messages = conversation.get("messages", [])
                if not isinstance(messages, list):
                    continue
                for index, item in enumerate(messages):
                    if not isinstance(item, dict) or item.get("role") != "user":
                        continue
                    answer = messages[index + 1] if index + 1 < len(messages) and isinstance(messages[index + 1], dict) else {}
                    rows.append(
                        {
                            "session_id": str(session_id)[:12],
                            "conversation": str(conversation.get("title", "Chat")),
                            "question": str(item.get("text", "")),
                            "answer": str(answer.get("text", "")),
                            "confidence": answer.get("confidence"),
                            "confidence_label": answer.get("confidence_label"),
                            "updatedAt": conversation.get("updatedAt", 0),
                        }
                    )
        rows.sort(key=lambda item: int(item.get("updatedAt") or 0), reverse=True)
        return rows[:80]

    def _admin_overview_payload(self) -> dict[str, object]:
        return {
            "status": self._status_payload(),
            "reports": list(reversed(self._load_reports()[-100:])),
            "user_count": len(self._load_json_dict(self.users_path)),
            "server_chat_count": sum(
                len(value) for value in self._load_json_dict(self.chats_path).values() if isinstance(value, list)
            ),
            "memory_notes": self.bot.list_memory_notes()[-80:],
            "memory_sources": self.bot.list_memory_sources(),
            "preferences": self.bot.list_preferences(),
            "subjects": self.bot.list_subjects(),
            "subject_briefs": self.bot.list_subject_briefs(),
            "recent_questions": self._recent_server_questions(),
            "low_confidence": [
                item for item in self._recent_server_questions()
                if str(item.get("confidence_label", "")).lower() == "faible"
            ][:30],
            "history": [
                {"user": turn.user, "assistant": turn.assistant}
                for turn in self.bot.history[-30:]
            ],
        }

    def _confidence_payload(self, message: str, answer: str) -> dict[str, object]:
        message_n = " ".join(message.lower().split())
        answer_n = " ".join(answer.lower().split())
        confidence = 0.58
        reasons: list[str] = []
        if any(char.isdigit() for char in message) and any(op in message for op in ("+", "-", "*", "/", "x", "÷")):
            confidence = 0.92
            reasons.append("calcul")
        if any(word in answer_n for word in ("document", "fiche", "base de connaissances", "d'apres", "d'après")):
            confidence = max(confidence, 0.82)
            reasons.append("document")
        if self.bot._detect_subject(message) or self.bot.last_subject:
            confidence = max(confidence, 0.68)
            reasons.append("contexte")
        if self.bot._is_vague_question(message) and not reasons:
            confidence = 0.38
            reasons.append("question vague")
        uncertainty = (
            "je ne sais pas",
            "pas sur",
            "pas sûr",
            "a verifier",
            "je peux me tromper",
            "question vague",
            "precise",
            "précise",
        )
        if any(part in answer_n for part in uncertainty):
            confidence = min(confidence, 0.48)
            reasons.append("incertain")
        if len(answer_n.split()) < 6:
            confidence = min(confidence, 0.55)
        label = "haute" if confidence >= 0.78 else "moyenne" if confidence >= 0.52 else "faible"
        return {
            "confidence": round(confidence, 2),
            "confidence_label": label,
            "confidence_reasons": reasons[:4],
        }

    def _save_memory_soon(self, force: bool = False) -> None:
        now = time.time()
        type(self).pending_memory_saves += 1
        if not force and type(self).pending_memory_saves < 8 and now - type(self).last_memory_save_at < 60:
            return
        self.bot.save()
        type(self).last_memory_save_at = now
        type(self).pending_memory_saves = 0

    def _session_id(self) -> str:
        cookie = self.headers.get("Cookie", "")
        for part in cookie.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "lucie_session" and value:
                return value[:96]
        session_id = secrets.token_urlsafe(24)
        self._set_session_cookie = f"lucie_session={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
        return session_id

    def _public_base_url(self) -> str:
        configured = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
        if configured:
            return configured
        proto = self.headers.get("X-Forwarded-Proto", "http").split(",", 1)[0].strip() or "http"
        host = self.headers.get("Host", "127.0.0.1:8000").strip()
        return f"{proto}://{host}"

    def _email_provider_config(self, provider: str) -> dict[str, object]:
        base_url = self._public_base_url()
        if provider == "gmail":
            return {
                "name": "gmail",
                "label": "Gmail",
                "client_id": os.getenv("GMAIL_CLIENT_ID", "").strip(),
                "client_secret": os.getenv("GMAIL_CLIENT_SECRET", "").strip(),
                "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "redirect_uri": f"{base_url}/oauth/gmail/callback",
                "scope": "https://www.googleapis.com/auth/gmail.readonly",
            }
        if provider == "outlook":
            return {
                "name": "outlook",
                "label": "Outlook",
                "client_id": os.getenv("OUTLOOK_CLIENT_ID", "").strip(),
                "client_secret": os.getenv("OUTLOOK_CLIENT_SECRET", "").strip(),
                "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "redirect_uri": f"{base_url}/oauth/outlook/callback",
                "scope": "openid offline_access User.Read Mail.Read",
            }
        raise ValueError("Provider mail inconnu.")

    def _email_token_key(self, provider: str, session_id: str | None = None) -> str:
        return f"{session_id or self._session_id()}:{provider}"

    def _email_status_payload(self) -> dict[str, object]:
        session_id = self._session_id()
        providers: dict[str, object] = {}
        for provider in ("gmail", "outlook"):
            config = self._email_provider_config(provider)
            token = self.email_tokens.get(self._email_token_key(provider, session_id), {})
            providers[provider] = {
                "label": config["label"],
                "configured": bool(config["client_id"]),
                "connected": bool(token.get("access_token")),
                "scope": config["scope"],
            }
        return {"ok": True, "providers": providers}

    def _handle_email_connect(self, provider: str) -> None:
        try:
            config = self._email_provider_config(provider)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if not config["client_id"]:
            self._send_html(
                "<!doctype html><meta charset='utf-8'><title>Lucie mail</title>"
                "<body style='font-family:Arial;padding:32px'>"
                f"<h1>{config['label']} pas encore configure</h1>"
                "<p>Ajoute les variables OAuth dans Render, puis reviens ici.</p>"
                "<p>Gmail: GMAIL_CLIENT_ID et GMAIL_CLIENT_SECRET. Outlook: OUTLOOK_CLIENT_ID et OUTLOOK_CLIENT_SECRET.</p>"
                "<p><a href='/'>Retour a Lucie</a></p></body>"
            )
            return
        session_id = self._session_id()
        state = secrets.token_urlsafe(24)
        self.email_states[state] = f"{session_id}:{provider}"
        params = {
            "client_id": config["client_id"],
            "redirect_uri": config["redirect_uri"],
            "response_type": "code",
            "scope": config["scope"],
            "state": state,
        }
        if provider == "gmail":
            params["access_type"] = "offline"
            params["prompt"] = "consent"
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", f"{config['auth_url']}?{urlencode(params)}")
        self._maybe_set_session_cookie()
        self.end_headers()

    def _handle_email_callback(self, provider: str) -> None:
        query = parse_qs(urlparse(self.path).query)
        if "error" in query:
            self._send_html(self._email_result_page("Connexion refusee", query["error"][0]))
            return
        code = query.get("code", [""])[0]
        state = query.get("state", [""])[0]
        state_value = self.email_states.pop(state, "")
        if not code or not state_value:
            self._send_html(self._email_result_page("Connexion impossible", "Le code OAuth est manquant ou expire."))
            return
        session_id, _, state_provider = state_value.partition(":")
        if state_provider != provider:
            self._send_html(self._email_result_page("Connexion impossible", "Le fournisseur ne correspond pas."))
            return
        config = self._email_provider_config(provider)
        if not config["client_secret"]:
            self._send_html(self._email_result_page("Secret OAuth manquant", "Ajoute le secret OAuth sur Render pour terminer la connexion."))
            return
        try:
            token = self._exchange_email_code(config, code)
        except Exception as exc:
            self._send_html(self._email_result_page("Connexion mail echouee", str(exc)))
            return
        token["connected_at"] = time.time()
        self.email_tokens[self._email_token_key(provider, session_id)] = token
        self._set_session_cookie = f"lucie_session={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
        self._send_html(self._email_result_page(f"{config['label']} connecte", "Lucie peut maintenant lire les derniers mails autorises."))

    def _exchange_email_code(self, config: dict[str, object], code: str) -> dict[str, object]:
        payload = urlencode(
            {
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "code": code,
                "redirect_uri": config["redirect_uri"],
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")
        request = Request(
            str(config["token_url"]),
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        if "access_token" not in data:
            raise RuntimeError("Google/Microsoft n'a pas renvoye de jeton d'acces.")
        return data if isinstance(data, dict) else {}

    def _email_result_page(self, title: str, message: str) -> str:
        return (
            "<!doctype html><meta charset='utf-8'><title>Lucie mail</title>"
            "<body style='font-family:Arial;padding:32px;line-height:1.5'>"
            f"<h1>{title}</h1><p>{message}</p><p><a href='/'>Retour a Lucie</a></p></body>"
        )

    def _handle_email_inbox(self) -> None:
        session_id = self._session_id()
        for provider in ("gmail", "outlook"):
            token = self.email_tokens.get(self._email_token_key(provider, session_id), {})
            if token.get("access_token"):
                try:
                    messages = self._fetch_email_messages(provider, str(token["access_token"]))
                except Exception as exc:
                    self._send_json({"error": f"Lecture {provider} impossible: {exc}"}, status=HTTPStatus.BAD_GATEWAY)
                    return
                self._send_json({"ok": True, "provider": provider, "messages": messages})
                return
        self._send_json(
            {"error": "Aucun compte mail connecte. Clique d'abord sur Gmail ou Outlook dans le menu."},
            status=HTTPStatus.BAD_REQUEST,
        )

    def _fetch_email_messages(self, provider: str, access_token: str) -> list[dict[str, str]]:
        if provider == "gmail":
            return self._fetch_gmail_messages(access_token)
        if provider == "outlook":
            return self._fetch_outlook_messages(access_token)
        return []

    def _api_get_json(self, url: str, access_token: str) -> dict[str, object]:
        request = Request(url, headers={"Authorization": f"Bearer {access_token}"})
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _fetch_gmail_messages(self, access_token: str) -> list[dict[str, str]]:
        listing = self._api_get_json(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults=5&q=newer_than:30d",
            access_token,
        )
        messages: list[dict[str, str]] = []
        for item in listing.get("messages", [])[:5] if isinstance(listing.get("messages"), list) else []:
            message_id = item.get("id") if isinstance(item, dict) else ""
            if not message_id:
                continue
            detail = self._api_get_json(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}?format=metadata&metadataHeaders=Subject&metadataHeaders=From&metadataHeaders=Date",
                access_token,
            )
            headers = {
                header.get("name", "").lower(): header.get("value", "")
                for header in detail.get("payload", {}).get("headers", [])
                if isinstance(header, dict)
            }
            messages.append(
                {
                    "from": headers.get("from", ""),
                    "subject": headers.get("subject", ""),
                    "date": headers.get("date", ""),
                    "preview": str(detail.get("snippet", ""))[:180],
                }
            )
        return messages

    def _fetch_outlook_messages(self, access_token: str) -> list[dict[str, str]]:
        data = self._api_get_json(
            "https://graph.microsoft.com/v1.0/me/messages?$top=5&$select=subject,from,receivedDateTime,bodyPreview",
            access_token,
        )
        messages: list[dict[str, str]] = []
        for item in data.get("value", [])[:5] if isinstance(data.get("value"), list) else []:
            if not isinstance(item, dict):
                continue
            sender = item.get("from", {}).get("emailAddress", {}) if isinstance(item.get("from"), dict) else {}
            messages.append(
                {
                    "from": sender.get("address", "") or sender.get("name", ""),
                    "subject": str(item.get("subject", "")),
                    "date": str(item.get("receivedDateTime", "")),
                    "preview": str(item.get("bodyPreview", ""))[:180],
                }
            )
        return messages

    def _calendar_provider_config(self, provider: str) -> dict[str, object]:
        base_url = self._public_base_url()
        if provider == "google":
            return {
                "name": "google",
                "label": "Google Calendar",
                "client_id": os.getenv("GOOGLE_CALENDAR_CLIENT_ID", "").strip()
                or os.getenv("GMAIL_CLIENT_ID", "").strip(),
                "client_secret": os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET", "").strip()
                or os.getenv("GMAIL_CLIENT_SECRET", "").strip(),
                "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "redirect_uri": f"{base_url}/oauth/calendar/google/callback",
                "scope": "https://www.googleapis.com/auth/calendar.events.readonly",
            }
        if provider == "outlook":
            return {
                "name": "outlook",
                "label": "Outlook Calendar",
                "client_id": os.getenv("OUTLOOK_CALENDAR_CLIENT_ID", "").strip()
                or os.getenv("OUTLOOK_CLIENT_ID", "").strip(),
                "client_secret": os.getenv("OUTLOOK_CALENDAR_CLIENT_SECRET", "").strip()
                or os.getenv("OUTLOOK_CLIENT_SECRET", "").strip(),
                "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "redirect_uri": f"{base_url}/oauth/calendar/outlook/callback",
                "scope": "openid offline_access User.Read Calendars.Read",
            }
        raise ValueError("Provider calendrier inconnu.")

    def _account_provider_config(self, provider: str) -> dict[str, object]:
        base_url = self._public_base_url()
        if provider == "google":
            return {
                "name": "google",
                "label": "Google",
                "mail_provider": "gmail",
                "calendar_provider": "google",
                "client_id": os.getenv("GOOGLE_ACCOUNT_CLIENT_ID", "").strip()
                or os.getenv("GMAIL_CLIENT_ID", "").strip()
                or os.getenv("GOOGLE_CALENDAR_CLIENT_ID", "").strip(),
                "client_secret": os.getenv("GOOGLE_ACCOUNT_CLIENT_SECRET", "").strip()
                or os.getenv("GMAIL_CLIENT_SECRET", "").strip()
                or os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET", "").strip(),
                "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "redirect_uri": f"{base_url}/oauth/account/google/callback",
                "scope": "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/calendar.events.readonly",
            }
        if provider == "microsoft":
            return {
                "name": "microsoft",
                "label": "Microsoft",
                "mail_provider": "outlook",
                "calendar_provider": "outlook",
                "client_id": os.getenv("MICROSOFT_ACCOUNT_CLIENT_ID", "").strip()
                or os.getenv("OUTLOOK_CLIENT_ID", "").strip()
                or os.getenv("OUTLOOK_CALENDAR_CLIENT_ID", "").strip(),
                "client_secret": os.getenv("MICROSOFT_ACCOUNT_CLIENT_SECRET", "").strip()
                or os.getenv("OUTLOOK_CLIENT_SECRET", "").strip()
                or os.getenv("OUTLOOK_CALENDAR_CLIENT_SECRET", "").strip(),
                "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "redirect_uri": f"{base_url}/oauth/account/microsoft/callback",
                "scope": "openid offline_access User.Read Mail.Read Calendars.Read",
            }
        raise ValueError("Compte inconnu.")

    def _handle_account_connect(self, provider: str) -> None:
        try:
            config = self._account_provider_config(provider)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if not config["client_id"]:
            self._send_html(
                "<!doctype html><meta charset='utf-8'><title>Lucie connexion</title>"
                "<body style='font-family:Arial;padding:32px;line-height:1.5'>"
                f"<h1>{config['label']} pas encore configure</h1>"
                "<p>Pour les utilisateurs ce sera simple: un bouton, puis ils valident chez Google ou Microsoft.</p>"
                "<p>Pour l'activer, ajoute les cles OAuth dans Render.</p>"
                "<p>Google: GOOGLE_ACCOUNT_CLIENT_ID et GOOGLE_ACCOUNT_CLIENT_SECRET "
                "(ou reutilise GMAIL_CLIENT_ID/GMAIL_CLIENT_SECRET). Microsoft: MICROSOFT_ACCOUNT_CLIENT_ID "
                "et MICROSOFT_ACCOUNT_CLIENT_SECRET (ou reutilise OUTLOOK_CLIENT_ID/OUTLOOK_CLIENT_SECRET).</p>"
                "<p><a href='/'>Retour a Lucie</a></p></body>"
            )
            return
        session_id = self._session_id()
        state = secrets.token_urlsafe(24)
        self.account_states[state] = f"{session_id}:{provider}"
        params = {
            "client_id": config["client_id"],
            "redirect_uri": config["redirect_uri"],
            "response_type": "code",
            "scope": config["scope"],
            "state": state,
        }
        if provider == "google":
            params["access_type"] = "offline"
            params["prompt"] = "consent"
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", f"{config['auth_url']}?{urlencode(params)}")
        self._maybe_set_session_cookie()
        self.end_headers()

    def _handle_account_callback(self, provider: str) -> None:
        query = parse_qs(urlparse(self.path).query)
        if "error" in query:
            self._send_html(self._email_result_page("Connexion refusee", query["error"][0]))
            return
        code = query.get("code", [""])[0]
        state = query.get("state", [""])[0]
        state_value = self.account_states.pop(state, "")
        if not code or not state_value:
            self._send_html(self._email_result_page("Connexion impossible", "Le code OAuth est manquant ou expire."))
            return
        session_id, _, state_provider = state_value.partition(":")
        if state_provider != provider:
            self._send_html(self._email_result_page("Connexion impossible", "Le fournisseur ne correspond pas."))
            return
        config = self._account_provider_config(provider)
        if not config["client_secret"]:
            self._send_html(self._email_result_page("Secret OAuth manquant", "Ajoute le secret OAuth sur Render pour terminer la connexion."))
            return
        try:
            token = self._exchange_email_code(config, code)
        except Exception as exc:
            self._send_html(self._email_result_page("Connexion echouee", str(exc)))
            return
        token["connected_at"] = time.time()
        self.email_tokens[self._email_token_key(str(config["mail_provider"]), session_id)] = dict(token)
        self.calendar_tokens[self._calendar_token_key(str(config["calendar_provider"]), session_id)] = dict(token)
        self._set_session_cookie = f"lucie_session={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
        self._send_html(
            self._email_result_page(
                f"{config['label']} connecte",
                "Lucie peut maintenant aider avec les mails et le calendrier autorises.",
            )
        )

    def _calendar_token_key(self, provider: str, session_id: str | None = None) -> str:
        return f"{session_id or self._session_id()}:{provider}"

    def _calendar_status_payload(self) -> dict[str, object]:
        session_id = self._session_id()
        providers: dict[str, object] = {}
        for provider in ("google", "outlook"):
            config = self._calendar_provider_config(provider)
            token = self.calendar_tokens.get(self._calendar_token_key(provider, session_id), {})
            providers[provider] = {
                "label": config["label"],
                "configured": bool(config["client_id"]),
                "connected": bool(token.get("access_token")),
                "scope": config["scope"],
            }
        return {"ok": True, "providers": providers}

    def _handle_calendar_connect(self, provider: str) -> None:
        try:
            config = self._calendar_provider_config(provider)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if not config["client_id"]:
            self._send_html(
                "<!doctype html><meta charset='utf-8'><title>Lucie calendrier</title>"
                "<body style='font-family:Arial;padding:32px'>"
                f"<h1>{config['label']} pas encore configure</h1>"
                "<p>Ajoute les variables OAuth dans Render, puis reviens ici.</p>"
                "<p>Google: GOOGLE_CALENDAR_CLIENT_ID et GOOGLE_CALENDAR_CLIENT_SECRET "
                "(ou reutilise GMAIL_CLIENT_ID/GMAIL_CLIENT_SECRET). Outlook: OUTLOOK_CALENDAR_CLIENT_ID "
                "et OUTLOOK_CALENDAR_CLIENT_SECRET.</p>"
                "<p><a href='/'>Retour a Lucie</a></p></body>"
            )
            return
        session_id = self._session_id()
        state = secrets.token_urlsafe(24)
        self.calendar_states[state] = f"{session_id}:{provider}"
        params = {
            "client_id": config["client_id"],
            "redirect_uri": config["redirect_uri"],
            "response_type": "code",
            "scope": config["scope"],
            "state": state,
        }
        if provider == "google":
            params["access_type"] = "offline"
            params["prompt"] = "consent"
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", f"{config['auth_url']}?{urlencode(params)}")
        self._maybe_set_session_cookie()
        self.end_headers()

    def _handle_calendar_callback(self, provider: str) -> None:
        query = parse_qs(urlparse(self.path).query)
        if "error" in query:
            self._send_html(self._email_result_page("Connexion calendrier refusee", query["error"][0]))
            return
        code = query.get("code", [""])[0]
        state = query.get("state", [""])[0]
        state_value = self.calendar_states.pop(state, "")
        if not code or not state_value:
            self._send_html(self._email_result_page("Connexion impossible", "Le code OAuth est manquant ou expire."))
            return
        session_id, _, state_provider = state_value.partition(":")
        if state_provider != provider:
            self._send_html(self._email_result_page("Connexion impossible", "Le fournisseur ne correspond pas."))
            return
        config = self._calendar_provider_config(provider)
        if not config["client_secret"]:
            self._send_html(self._email_result_page("Secret OAuth manquant", "Ajoute le secret OAuth sur Render pour terminer la connexion."))
            return
        try:
            token = self._exchange_email_code(config, code)
        except Exception as exc:
            self._send_html(self._email_result_page("Connexion calendrier echouee", str(exc)))
            return
        token["connected_at"] = time.time()
        self.calendar_tokens[self._calendar_token_key(provider, session_id)] = token
        self._set_session_cookie = f"lucie_session={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
        self._send_html(self._email_result_page(f"{config['label']} connecte", "Lucie peut maintenant lire les prochains evenements autorises."))

    def _handle_calendar_events(self) -> None:
        session_id = self._session_id()
        for provider in ("google", "outlook"):
            token = self.calendar_tokens.get(self._calendar_token_key(provider, session_id), {})
            if token.get("access_token"):
                try:
                    events = self._fetch_calendar_events(provider, str(token["access_token"]))
                except Exception as exc:
                    self._send_json({"error": f"Lecture calendrier {provider} impossible: {exc}"}, status=HTTPStatus.BAD_GATEWAY)
                    return
                self._send_json({"ok": True, "provider": provider, "events": events})
                return
        self._send_json(
            {"error": "Aucun calendrier connecte. Clique d'abord sur Google ou Outlook dans le menu Calendrier."},
            status=HTTPStatus.BAD_REQUEST,
        )

    def _fetch_calendar_events(self, provider: str, access_token: str) -> list[dict[str, str]]:
        if provider == "google":
            return self._fetch_google_calendar_events(access_token)
        if provider == "outlook":
            return self._fetch_outlook_calendar_events(access_token)
        return []

    def _fetch_google_calendar_events(self, access_token: str) -> list[dict[str, str]]:
        now = datetime.now(timezone.utc)
        time_min = now.isoformat().replace("+00:00", "Z")
        query = urlencode(
            {
                "maxResults": "10",
                "singleEvents": "true",
                "orderBy": "startTime",
                "timeMin": time_min,
            }
        )
        data = self._api_get_json(
            f"https://www.googleapis.com/calendar/v3/calendars/primary/events?{query}",
            access_token,
        )
        events: list[dict[str, str]] = []
        for item in data.get("items", [])[:10] if isinstance(data.get("items"), list) else []:
            if not isinstance(item, dict):
                continue
            start = item.get("start", {}) if isinstance(item.get("start"), dict) else {}
            events.append(
                {
                    "title": str(item.get("summary", "")),
                    "start": str(start.get("dateTime") or start.get("date") or ""),
                    "end": str((item.get("end", {}) if isinstance(item.get("end"), dict) else {}).get("dateTime", "")),
                    "location": str(item.get("location", "")),
                }
            )
        return events

    def _fetch_outlook_calendar_events(self, access_token: str) -> list[dict[str, str]]:
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=30)
        query = urlencode(
            {
                "startDateTime": now.isoformat(),
                "endDateTime": end.isoformat(),
                "$top": "10",
                "$select": "subject,organizer,start,end,location",
                "$orderby": "start/dateTime",
            }
        )
        data = self._api_get_json(
            f"https://graph.microsoft.com/v1.0/me/calendarView?{query}",
            access_token,
        )
        events: list[dict[str, str]] = []
        for item in data.get("value", [])[:10] if isinstance(data.get("value"), list) else []:
            if not isinstance(item, dict):
                continue
            start = item.get("start", {}) if isinstance(item.get("start"), dict) else {}
            end_value = item.get("end", {}) if isinstance(item.get("end"), dict) else {}
            location = item.get("location", {}) if isinstance(item.get("location"), dict) else {}
            events.append(
                {
                    "title": str(item.get("subject", "")),
                    "start": str(start.get("dateTime", "")),
                    "end": str(end_value.get("dateTime", "")),
                    "location": str(location.get("displayName", "")),
                }
            )
        return events

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._session_id()
            self._send_html(HTML_PAGE.replace("__IA_API_KEY__", self.api_key))
            return
        if path == "/admin":
            self._session_id()
            self._send_html(ADMIN_PAGE)
            return
        if path == "/health":
            self._send_json({"ok": True, "message": "Lucie est en ligne."})
            return
        if path == "/robots.txt":
            self._send_text("User-agent: *\nAllow: /\nSitemap: /sitemap.xml\n", "text/plain; charset=utf-8")
            return
        if path == "/sitemap.xml":
            self._send_text(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n"
                "  <url><loc>/</loc></url>\n"
                "</urlset>\n",
                "application/xml; charset=utf-8",
            )
            return
        if path == "/api/status":
            if not self._authorized():
                self._send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return
            self._send_json(self._status_payload())
            return
        if path == "/api/admin/overview":
            if not self._authorized_admin():
                self._send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return
            self._send_json(self._admin_overview_payload())
            return
        if path == "/api/profile":
            self._send_json({"ok": True, "profile": self._current_profile()})
            return
        if path == "/api/conversations":
            self._send_json({"ok": True, "conversations": self._load_session_chats()})
            return
        if path == "/api/search":
            query = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
            snippets = self.bot.predict_documents(query)[:8] if query else []
            self._send_json(
                {
                    "ok": True,
                    "results": [
                        {"content": item.content, "score": item.score, "source": item.source}
                        for item in snippets
                    ],
                }
            )
            return
        if path == "/api/email/status":
            self._send_json(self._email_status_payload())
            return
        if path == "/api/email/inbox":
            self._handle_email_inbox()
            return
        if path == "/api/calendar/status":
            self._send_json(self._calendar_status_payload())
            return
        if path == "/api/calendar/events":
            self._handle_calendar_events()
            return
        if path in {"/connect/gmail", "/connect/outlook"}:
            self._handle_email_connect(path.rsplit("/", 1)[-1])
            return
        if path in {"/connect/calendar/google", "/connect/calendar/outlook"}:
            self._handle_calendar_connect(path.rsplit("/", 1)[-1])
            return
        if path in {"/connect/account/google", "/connect/account/microsoft"}:
            self._handle_account_connect(path.rsplit("/", 1)[-1])
            return
        if path in {"/oauth/gmail/callback", "/oauth/outlook/callback"}:
            provider = path.split("/")[2]
            self._handle_email_callback(provider)
            return
        if path in {"/oauth/calendar/google/callback", "/oauth/calendar/outlook/callback"}:
            provider = path.split("/")[3]
            self._handle_calendar_callback(provider)
            return
        if path in {"/oauth/account/google/callback", "/oauth/account/microsoft/callback"}:
            provider = path.split("/")[3]
            self._handle_account_callback(provider)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_json_body()

        if path == "/api/chat":
            self._handle_chat(body)
            return
        if path == "/api/teach":
            if not self._authorized():
                self._send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return
            self._handle_teach(body)
            return
        if path == "/api/document":
            if not self._authorized():
                self._send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return
            self._handle_document(body)
            return
        if path == "/api/report":
            self._handle_report(body)
            return
        if path == "/api/profile":
            profile = body.get("profile", body)
            if not isinstance(profile, dict):
                self._send_json({"error": "Profil invalide"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, "profile": self._save_profile(profile)})
            return
        if path == "/api/admin/compact-memory":
            if not self._authorized_admin():
                self._send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return
            stats = self.bot.compact_memory()
            type(self).last_memory_save_at = time.time()
            type(self).pending_memory_saves = 0
            self._send_json({"ok": True, "stats": stats})
            return
        if path == "/api/admin/delete-memory-note":
            if not self._authorized_admin():
                self._send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
                return
            note = str(body.get("note", "")).strip()
            if note:
                self.bot.memory_notes = [item for item in self.bot.memory_notes if item != note]
                for key, values in list(self.bot.memory_sources.items()):
                    self.bot.memory_sources[key] = [item for item in values if item != note]
                self.bot.conversation_summary = self.bot._build_conversation_summary()
                self._save_memory_soon(force=True)
            self._send_json({"ok": True, "memory_count": len(self.bot.memory_notes)})
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_chat(self, body: dict[str, object]) -> None:
        message = str(body.get("message", "")).strip()
        if not message:
            self._send_json({"error": "Message requis"}, status=HTTPStatus.BAD_REQUEST)
            return
        profile = body.get("profile")
        profile_hint = ""
        if isinstance(profile, dict):
            parts = []
            name = str(profile.get("name", "")).strip()
            style = str(profile.get("style", "")).strip()
            goal = str(profile.get("goal", "")).strip()
            topics = str(profile.get("topics", "")).strip()
            if name:
                parts.append(f"prenom={name}")
            if style:
                parts.append(f"style={style}")
            if goal:
                parts.append(f"objectif={goal}")
            if topics:
                parts.append(f"sujets={topics}")
            if parts:
                profile_hint = "Profil utilisateur: " + "; ".join(parts)
            for key, value in {
                "prenom": name,
                "style": style,
                "objectif": goal,
                "sujets": topics,
            }.items():
                if value:
                    self.bot.remember_preference(key, value)
            if any((name, style, goal, topics)):
                self._save_profile(profile)
        conversation_id = str(body.get("conversation_id", "")).strip()
        conversation_title = str(body.get("conversation_title", "")).strip()

        try:
            if message.startswith("/teach"):
                question, answer = self._parse_teach_command(message)
                self.bot.teach(question, answer)
                self._save_memory_soon(force=True)
                self._send_json(
                    {
                        "answer": "Bien recu, j'ai appris ca.",
                        "note": f"Enregistre: {question}",
                        "mode": "local",
                        "model": self.bot.model,
                        "intent": None,
                        "entities": [],
                        "knowledge": [],
                        "memory_sources_count": len(self.bot.memory_sources),
                        "pending_action": self.bot.pending_action,
                    }
                )
                return
            bot_message = f"{profile_hint}\n\nQuestion: {message}" if profile_hint else message
            answer_text = self.bot.answer(bot_message)
            if not self.bot.api_available:
                subject = self.bot._detect_subject(message)
                self.bot._remember(message, answer_text)
                self.bot._remember_subject(subject, message, answer_text)
            self._save_memory_soon()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            self._send_json(
                {"error": f"Erreur de sauvegarde: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        confidence = self._confidence_payload(message, answer_text)
        if confidence["confidence"] < 0.45:
            answer_text = (
                "Je ne suis pas assez sure pour repondre proprement. "
                "Peux-tu preciser le sujet, ou me donner une phrase de contexte ?"
            )
            confidence = {"confidence": 0.32, "confidence_label": "faible", "confidence_reasons": ["incertain"]}
        if conversation_id:
            try:
                self._save_chat_turn(conversation_id, conversation_title, message, answer_text, confidence)
            except OSError:
                pass
        self._send_json(
            {
                "answer": answer_text,
                "note": "",
                "mode": "openai" if self.bot.api_available else "local",
                "model": self.bot.model,
                "memory_count": len(self.bot.memory_notes),
                "memory_sources_count": len(self.bot.memory_sources),
                "preferences_count": len(self.bot.preferences),
                "subjects_count": len(self.bot.subject_memory),
                "document_count": len(self.bot.documents),
                "example_count": self.bot.total_example_count(),
                "last_subject": self.bot.last_subject,
                "conversation_summary": self.bot.get_conversation_summary(),
                "subject_briefs": self.bot.list_subject_briefs(),
                "robot_status": self._robot_status_payload(),
                "intent": None,
                "entities": [],
                "knowledge": [],
                "pending_action": self.bot.pending_action,
                **confidence,
            }
        )

    def _parse_teach_command(self, message: str) -> tuple[str, str]:
        payload = message[len("/teach") :].strip()
        if "|" not in payload:
            raise ValueError("Format attendu: /teach question | reponse")
        question, answer = [part.strip() for part in payload.split("|", 1)]
        if not question or not answer:
            raise ValueError("Question et reponse ne doivent pas etre vides.")
        return question, answer

    def _handle_teach(self, body: dict[str, object]) -> None:
        question = str(body.get("question", "")).strip()
        answer = str(body.get("answer", "")).strip()
        if not question or not answer:
            self._send_json(
                {"error": "Question et reponse requises"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            self.bot.teach(question, answer)
            self._save_memory_soon(force=True)
        except OSError as exc:
            self._send_json(
                {"error": f"Erreur de sauvegarde: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json({"answer": "Bien recu, j'ai appris ca."})

    def _handle_document(self, body: dict[str, object]) -> None:
        title = str(body.get("title", "")).strip()
        content = str(body.get("content", "")).strip()
        if not content:
            self._send_json({"error": "Contenu du document requis"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            self.bot.add_document(title, content)
            self._save_memory_soon(force=True)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            self._send_json(
                {"error": f"Erreur de sauvegarde: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        document_title = title or f"Document {len(self.bot.documents)}"
        self._send_json(
            {
                "answer": f"Document ajoute: {document_title}",
                "document_count": len(self.bot.documents),
            }
        )

    def _handle_report(self, body: dict[str, object]) -> None:
        question = str(body.get("question", "")).strip()
        answer = str(body.get("answer", "")).strip()
        correction = str(body.get("correction", "")).strip()
        title = str(body.get("title", "Discussion Lucie")).strip()[:140]
        if not question:
            self._send_json({"error": "Question requise"}, status=HTTPStatus.BAD_REQUEST)
            return
        profile = body.get("profile")
        if not isinstance(profile, dict):
            profile = {}
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id(),
            "title": title or "Discussion Lucie",
            "question": question[:3000],
            "answer": answer[:5000],
            "correction": correction[:3000],
            "profile": profile,
            "url": str(body.get("url", ""))[:500],
            "user_agent": self.headers.get("User-Agent", "")[:300],
        }
        try:
            reports = self._save_report(report)
        except OSError as exc:
            self._send_json(
                {"error": f"Erreur de sauvegarde: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json({"ok": True, "message": "Signalement recu", "report_count": len(reports)})

    def _read_json_body(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _authorized(self) -> bool:
        host = (self.headers.get("Host", "") or "").split(":", 1)[0].strip().lower()
        client_host = self.client_address[0] if self.client_address else ""
        if host in {"127.0.0.1", "localhost"} and client_host in {"127.0.0.1", "::1"}:
            return True
        if self._authorized_admin():
            return True
        if not self.api_key:
            return True
        header = self.headers.get("Authorization", "").strip()
        return header == f"Bearer {self.api_key}"

    def _authorized_admin(self) -> bool:
        code = self.headers.get("X-Admin-Code", "").strip()
        if code and secrets.compare_digest(code, self.admin_code):
            return True
        header = self.headers.get("Authorization", "").strip()
        return bool(self.api_key and header == f"Bearer {self.api_key}")

    def _maybe_set_session_cookie(self) -> None:
        cookie = getattr(self, "_set_session_cookie", "")
        if cookie:
            self.send_header("Set-Cookie", cookie)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, content_type: str) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(data)

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(data)


def run_web_server(host: str, port: int, memory_path: Path) -> None:
    try:
        bot = LearningBot.load(memory_path)
    except Exception as exc:
        raise RuntimeError(f"Erreur au demarrage: {exc}") from exc

    AppHandler.bot = bot
    api_key_file = memory_path.parent / "ia_api_key.txt"
    api_key = os.getenv("IA_API_KEY", "").strip()
    if not api_key:
        if api_key_file.exists():
            api_key = api_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            api_key = secrets.token_urlsafe(32)
            api_key_file.write_text(api_key + "\n", encoding="utf-8")
    AppHandler.api_key = api_key
    AppHandler.admin_code = os.getenv("ADMIN_ACCESS_CODE", "042724").strip() or "042724"
    AppHandler.last_memory_save_at = time.time()
    AppHandler.pending_memory_saves = 0
    try:
        server = ThreadingHTTPServer((host, port), AppHandler)
    except OSError as exc:
        raise RuntimeError(
            f"Impossible de demarrer le serveur sur {host}:{port}. "
            f"Le port est peut-etre deja pris. Erreur: {exc}"
        ) from exc

    print(f"Interface web: http://{host}:{port}")
    print(f"Cle API locale enregistree dans {api_key_file}")
    if bot.startup_warning:
        print(f"Avertissement: {bot.startup_warning}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServeur arrete.")
    finally:
        server.server_close()
