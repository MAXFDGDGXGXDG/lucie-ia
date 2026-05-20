from __future__ import annotations

import json
import os
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

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
    .voice-tools {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .voice-tools button {
      border: 1px solid var(--border);
      background: rgba(2, 6, 23, 0.55);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
    }
    .voice-tools button[data-active="1"] {
      border-color: rgba(56, 189, 248, 0.48);
      background: rgba(56, 189, 248, 0.14);
    }
    .voice-state {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
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
    .hint,
    .voice-state {
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
    .chips button,
    .voice-tools button {
      background: #ffffff;
      color: #374151;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .chips button:hover,
    .voice-tools button:hover {
      transform: translateY(-1px);
      background: #f9fafb;
      border-color: #d1d5db;
    }
    .voice-tools button[data-active="1"] {
      background: #ecfdf5;
      border-color: #a7f3d0;
      color: #047857;
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
    .voice-tools,
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
      grid-template-columns: 280px minmax(0, 1fr);
    }
    body.sidebar-closed .workspace {
      grid-template-columns: 1fr;
    }
    .chat-sidebar {
      height: calc(100vh - 64px);
      border-right: 1px solid var(--border);
      background: #f7f7f8;
      padding: 14px;
      overflow: auto;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }
    body.sidebar-closed .chat-sidebar {
      display: none;
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
    }
    .chat-head {
      padding-top: 36px;
    }
    .chips,
    .voice-tools,
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
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <button class="sidebar-toggle" id="sidebar-toggle" type="button" aria-label="Ouvrir ou fermer les discussions">☰</button>
      <div class="brand">
        <h1>Lucie</h1>
        <p>Assistant personnel</p>
      </div>
      <div id="status" class="status"><span class="dot"></span><span>En attente...</span></div>
    </header>

    <main class="stage">
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
          <div class="voice-tools">
            <button type="button" id="voice-toggle">Parler</button>
            <button type="button" id="voice-speak">Voix ON</button>
            <span class="voice-state" id="voice-state">Voix prete.</span>
          </div>
          <div class="hint" id="hint">Prete a repondre.</div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const API_KEY = "__IA_API_KEY__";
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message");
    const hint = document.getElementById("hint");
    const status = document.getElementById("status");
    const chatLog = document.getElementById("chat-log");
    const voiceToggle = document.getElementById("voice-toggle");
    const voiceSpeakToggle = document.getElementById("voice-speak");
    const voiceState = document.getElementById("voice-state");
    const sidebarToggle = document.getElementById("sidebar-toggle");
    const stage = document.querySelector(".stage");
    const chatCard = document.querySelector(".chat-card");
    const chatSidebar = document.createElement("aside");
    chatSidebar.className = "chat-sidebar";
    chatSidebar.innerHTML = `
      <div class="sidebar-head">
        <div class="sidebar-line">
          <strong>Discussions</strong>
          <button class="sidebar-close" id="sidebar-close" type="button" aria-label="Fermer">×</button>
        </div>
        <button class="new-chat" id="new-chat" type="button">+ Nouveau chat</button>
        <button class="report-btn" id="report-bad-answer" type="button">Signaler la derniere reponse</button>
        <div class="report-note">Envoie la question et la reponse a Maxence pour ameliorer Lucie.</div>
      </div>
      <div class="chat-list" id="chat-list"></div>
    `;
    const sidebarClose = chatSidebar.querySelector("#sidebar-close");
    const newChatButton = chatSidebar.querySelector("#new-chat");
    const reportBadAnswerButton = chatSidebar.querySelector("#report-bad-answer");
    const chatList = chatSidebar.querySelector("#chat-list");
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

    if (stage && chatCard && !chatSidebar.isConnected) {
      const workspace = document.createElement("div");
      workspace.className = "workspace";
      stage.insertBefore(workspace, chatCard);
      workspace.appendChild(chatSidebar);
      workspace.appendChild(chatCard);
    }

    function escapeText(text) {
      return String(text || "");
    }

    const STORE_KEY = "lucie_conversations_v1";
    let conversations = [];
    let activeConversationId = "";

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
        addBubble("system", "Lucie est prete. Envoie un message pour commencer.", false);
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

    function reportBadAnswer() {
      const report = latestProblemReport();
      if (!report || !report.question) {
        alert("Il faut d'abord poser une question a Lucie.");
        return;
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
        "",
        "Envoye depuis l'app Lucie.",
      ].join("\n");
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

    function addBubble(role, text, save = true) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;
      bubble.textContent = escapeText(text);
      chatLog.appendChild(bubble);
      chatLog.scrollTop = chatLog.scrollHeight;
      if (save && role !== "system") {
        rememberMessage(role, text);
      }
      return bubble;
    }

    function ensureWelcome() {
      if (!chatLog.children.length) {
        addBubble("system", "Lucie est prete. Envoie un message pour commencer.");
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

    function setVoiceState(text, active = false) {
      if (voiceState) {
        voiceState.textContent = text;
      }
      if (voiceToggle) {
        voiceToggle.dataset.active = active ? "1" : "0";
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

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let listening = false;
    let autoSpeak = true;

    function refreshVoiceStatus() {
      if (voiceSpeakToggle) {
        voiceSpeakToggle.dataset.active = autoSpeak ? "1" : "0";
        voiceSpeakToggle.textContent = autoSpeak ? "Voix ON" : "Voix OFF";
      }
      if (!listening) {
        setVoiceState(autoSpeak ? "Voix prête." : "Voix coupée.", false);
      }
    }

    function speakText(text) {
      if (!autoSpeak || !("speechSynthesis" in window) || !window.SpeechSynthesisUtterance) {
        return;
      }
      const clean = String(text || "").trim().replace(/^Robot:\s*/i, "");
      if (!clean) return;
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(clean);
      utter.lang = "fr-FR";
      utter.rate = 1;
      utter.pitch = 1;
      const voices = window.speechSynthesis.getVoices().filter((voice) => String(voice.lang || "").toLowerCase().startsWith("fr"));
      if (voices.length) {
        utter.voice = voices[0];
      }
      utter.onstart = () => setVoiceState("Je parle...", true);
      utter.onend = () => refreshVoiceStatus();
      utter.onerror = () => setVoiceState("Voix indisponible.", false);
      window.speechSynthesis.speak(utter);
    }

    function ensureRecognition() {
      if (!SpeechRecognition) return null;
      if (recognition) return recognition;
      recognition = new SpeechRecognition();
      recognition.lang = "fr-FR";
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.onstart = () => {
        listening = true;
        setVoiceState("J'écoute...", true);
      };
      recognition.onresult = (event) => {
        let transcript = "";
        for (let index = 0; index < event.results.length; index += 1) {
          transcript += event.results[index][0].transcript;
        }
        const clean = transcript.trim();
        if (clean) {
          setVoiceState(`Entendu: ${shortText(clean, 60)}`, true);
        }
        const last = event.results[event.results.length - 1];
        if (last && last.isFinal && clean) {
          input.value = clean;
          form.requestSubmit();
        }
      };
      recognition.onerror = () => {
        listening = false;
        setVoiceState("Micro indisponible.", false);
      };
      recognition.onend = () => {
        listening = false;
        refreshVoiceStatus();
      };
      return recognition;
    }

    function toggleListening() {
      const current = ensureRecognition();
      if (!current) {
        setVoiceState("La reconnaissance vocale n'est pas disponible ici.", false);
        return;
      }
      if (listening) {
        current.stop();
        return;
      }
      try {
        current.start();
      } catch (error) {
        setVoiceState("Impossible de lancer le micro.", false);
      }
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

    if (sidebarToggle) {
      sidebarToggle.addEventListener("click", () => {
        document.body.classList.toggle("sidebar-closed");
      });
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

    if (voiceToggle) {
      voiceToggle.addEventListener("click", () => {
        if (!SpeechRecognition) {
          setVoiceState("Le navigateur ne gère pas le micro vocal.", false);
          return;
        }
        toggleListening();
      });
      voiceToggle.dataset.active = "0";
      if (!SpeechRecognition) {
        voiceToggle.disabled = true;
        voiceToggle.textContent = "Micro indisponible";
        setVoiceState("Micro vocal non pris en charge ici.", false);
      }
    }

    if (voiceSpeakToggle) {
      voiceSpeakToggle.addEventListener("click", () => {
        autoSpeak = !autoSpeak;
        refreshVoiceStatus();
        if (!autoSpeak && "speechSynthesis" in window) {
          window.speechSynthesis.cancel();
        }
      });
      if (!("speechSynthesis" in window)) {
        voiceSpeakToggle.disabled = true;
        voiceSpeakToggle.textContent = "Voix indisponible";
        setVoiceState("La synthèse vocale n'est pas disponible ici.", false);
      } else {
        refreshVoiceStatus();
        window.speechSynthesis.onvoiceschanged = refreshVoiceStatus;
      }
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
      hint.textContent = "Analyse en cours...";
      setStatus("Analyse", true);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({ message }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Requete refusee");
        }

        addBubble("assistant", data.answer || "OK");
        speakText(data.answer || "OK");

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
    input.focus();
    loadStatus();
  </script>
  <script>
    (() => {
      const form = document.getElementById("chat-form");
      const input = document.getElementById("message");
      const chatLog = document.getElementById("chat-log");
      const status = document.getElementById("status");
      if (!form || !input || !chatLog) return;

      function setStatusSafe(text) {
        if (status) {
          status.innerHTML = `<span class="dot"></span><span>${text}</span>`;
        }
      }

      function bubble(role, text) {
        const item = document.createElement("div");
        item.className = `bubble ${role}`;
        item.textContent = String(text || "");
        chatLog.appendChild(item);
        chatLog.scrollTop = chatLog.scrollHeight;
        return item;
      }

      async function sendMessage(event) {
        event.preventDefault();
        event.stopImmediatePropagation();
        const message = input.value.trim();
        if (!message) return;
        input.value = "";
        bubble("user", message);
        setStatusSafe("Analyse...");
        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || "Erreur serveur");
          }
          bubble("assistant", data.answer || "OK");
          setStatusSafe("Pret");
        } catch (error) {
          bubble("assistant", `Erreur: ${String(error.message || error)}`);
          setStatusSafe("Erreur");
        } finally {
          input.focus();
        }
      }

      form.addEventListener("submit", sendMessage, true);
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
        }
      }, true);
      setStatusSafe("Pret");
    })();
  </script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    bot: LearningBot
    api_key: str = ""

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML_PAGE.replace("__IA_API_KEY__", self.api_key))
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
            self._send_json(
                {
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
          "robot_status": self.bot.robot_bridge_status(),
        }
            )
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/") and not self._authorized():
            self._send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
            return
        body = self._read_json_body()

        if path == "/api/chat":
            self._handle_chat(body)
            return
        if path == "/api/teach":
            self._handle_teach(body)
            return
        if path == "/api/document":
            self._handle_document(body)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_chat(self, body: dict[str, object]) -> None:
        message = str(body.get("message", "")).strip()
        if not message:
            self._send_json({"error": "Message requis"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if message.startswith("/teach"):
                question, answer = self._parse_teach_command(message)
                self.bot.teach(question, answer)
                self.bot.save()
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
            subject = self.bot._detect_subject(message)
            prediction = self.bot.predict_intent(message)
            entities = self.bot.predict_entities(message)
            knowledge = self.bot.predict_knowledge(message)
            answer_text = self.bot.answer(message)
            if not self.bot.api_available:
                self.bot._remember(message, answer_text)
                self.bot._remember_subject(subject, message, answer_text)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            self._send_json(
                {"error": f"Erreur de sauvegarde: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

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
                "robot_status": self.bot.robot_bridge_status(),
                "intent": None
                if prediction is None
                else {
                    "label": prediction.label,
                    "confidence": round(prediction.confidence, 3),
                },
                "entities": []
                if entities is None
                else [
                    {
                        "start": entity.start,
                        "end": entity.end,
                        "text": entity.text,
                        "label": entity.label,
                    }
                    for entity in entities.entities
                ],
                "knowledge": []
                if not knowledge
                else [
                    {
                        "content": item.content,
                        "score": round(item.score, 3),
                        "source": item.source,
                    }
                    for item in knowledge
                ],
                "pending_action": self.bot.pending_action,
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
            self.bot.save()
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
            self.bot.save()
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
        if not self.api_key:
            return True
        header = self.headers.get("Authorization", "").strip()
        return header == f"Bearer {self.api_key}"

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, content_type: str) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
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
