"""
CatGirlWindow —— 猫娘主窗口（纯视图）
只负责 UI 渲染，业务逻辑全部通过 state_manager / worker 完成。
"""
import time

import pygame

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QFrame, QSizePolicy, QProgressBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QFont, QTextCursor

from config import (
    IMAGE_DIR, EXPRESSION_MAP, VALID_EMOTIONS,
    EMOTION_DECAY_INTERVAL_MS, EMOTION_DECAY_TIMEOUT_SEC,
    ACTIVE_CHAT_INTERVAL_MS, ACTIVE_CHAT_IDLE_SEC,
)


class CatGirlWindow(QMainWindow):
    """猫娘主窗口 —— 纯视图"""

    def __init__(self, state, worker, tts):
        """
        Args:
            state: NoraStateManager 实例
            worker: VoiceChatWorker 实例
            tts: TTSEngine 实例
        """
        super().__init__()
        self._state = state
        self._worker = worker
        self._tts = tts

        self._current_expression = state.current_emotion
        self._talking = False
        self._talk_frame = False
        self._active_chat_cooldown = False

        # 加载表情图片
        self._image_cache = {}
        self._load_images()

        # 构建 UI
        self._init_ui()

        # 连接信号
        self._connect_signals()

        # ---- 定时器 ----
        # 嘴部动画
        self._talk_timer = QTimer(self)
        self._talk_timer.timeout.connect(self._talk_animate)
        self._talk_timer.start(30)

        # 情绪衰减
        self._emotion_decay_timer = QTimer(self)
        self._emotion_decay_timer.timeout.connect(self._check_emotion_decay)
        self._emotion_decay_timer.start(EMOTION_DECAY_INTERVAL_MS)

        # 主动聊天
        self._active_chat_timer = QTimer(self)
        self._active_chat_timer.timeout.connect(self._check_active_chat)
        self._active_chat_timer.start(ACTIVE_CHAT_INTERVAL_MS)

    # ================================================================
    # 图片加载
    # ================================================================
    def _load_images(self):
        for expr, filename in EXPRESSION_MAP.items():
            import os
            filepath = os.path.join(IMAGE_DIR, filename)
            if os.path.exists(filepath):
                pix = QPixmap(filepath)
                if not pix.isNull():
                    self._image_cache[expr] = pix
                else:
                    print(f"警告：无法加载图片 {filepath}")
            else:
                print(f"警告：找不到图片 {filepath}")

        if not self._image_cache:
            raise RuntimeError("没有找到任何猫娘图片！")
        print(f"已加载 {len(self._image_cache)} 张猫娘表情图片\n")

    # ================================================================
    # UI 初始化
    # ================================================================
    def _init_ui(self):
        self.setWindowTitle(f"猫娘 Nora Plus - {self._state.personality_config['name']}")
        self.setMinimumSize(460, 700)
        self.resize(540, 820)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QTextEdit {
                background-color: #2a2a3e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel#statusLabel {
                color: #a0a0c0;
                font-size: 13px;
                padding: 4px 8px;
            }
            QLabel#favorLabel {
                color: #ff6b9d;
                font-size: 14px;
                font-weight: bold;
            }
            QProgressBar {
                background-color: #2a2a3e;
                border: 1px solid #555;
                border-radius: 6px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #ff6b9d;
                border-radius: 5px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ---- 猫娘图片 ----
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(280)
        self.image_label.setMaximumHeight(360)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Preferred)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #252540;
                border: 2px solid #ff6b9d;
                border-radius: 12px;
            }
        """)
        layout.addWidget(self.image_label)

        # ---- 好感度区域 ----
        favor_frame = QFrame()
        favor_frame.setStyleSheet("QFrame { background-color: #252540; border-radius: 8px; padding: 6px; }")
        favor_layout = QHBoxLayout(favor_frame)
        favor_layout.setContentsMargins(10, 6, 10, 6)
        favor_layout.setSpacing(10)

        fav = self._state.favorability
        self.favor_label = QLabel(f"❤️ 好感度  {fav} / 100")
        self.favor_label.setObjectName("favorLabel")
        favor_layout.addWidget(self.favor_label)

        self.favor_bar = QProgressBar()
        self.favor_bar.setMinimum(0)
        self.favor_bar.setMaximum(100)
        self.favor_bar.setValue(fav)
        self.favor_bar.setTextVisible(True)
        self.favor_bar.setFormat(f"{fav} / 100")
        self.favor_bar.setMinimumWidth(180)
        favor_layout.addWidget(self.favor_bar, stretch=1)
        layout.addWidget(favor_frame)

        # ---- 分隔线 ----
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("QFrame { color: #444; }")
        layout.addWidget(sep1)

        # ---- 聊天记录 ----
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 11))
        self.chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        layout.addWidget(self.chat_display, stretch=1)

        # ---- 分隔线 ----
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("QFrame { color: #444; }")
        layout.addWidget(sep2)

        # ---- 状态栏 ----
        self.status_label = QLabel("状态：初始化...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.status_label)

        # 初始显示
        self._apply_pixmap(self._state.current_emotion)

    # ================================================================
    # 信号连接
    # ================================================================
    def _connect_signals(self):
        w = self._worker
        w.status_signal.connect(self.on_status)
        w.expression_signal.connect(self._show_expression)
        w.user_msg_signal.connect(self.on_user_msg)
        w.catgirl_token_signal.connect(self.on_catgirl_token)
        w.catgirl_done_signal.connect(self.on_catgirl_done)
        w.request_close_signal.connect(self.on_request_close)
        w.favorability_signal.connect(self._update_favor_ui)
        w.favorability_change_signal.connect(self._on_favor_change)
        w.emotion_signal.connect(self._on_emotion_change)

    # ================================================================
    # 贴图 + 表情切换
    # ================================================================
    def _apply_pixmap(self, expr):
        pix = self._image_cache.get(expr)
        if pix is None:
            pix = self._image_cache.get("normal")
            if pix is None:
                return

        label_w = max(self.image_label.width(), 10) or 480
        label_h = max(self.image_label.height(), 10) or 350

        scaled = pix.scaled(label_w - 20, label_h - 10,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _show_expression(self, expr):
        if expr not in VALID_EMOTIONS and expr != "talk":
            expr = "normal"
        self._current_expression = expr
        if not self._talking:
            self._apply_pixmap(expr)

    # ================================================================
    # 嘴部动画
    # ================================================================
    def _talk_animate(self):
        pos = self._tts.get_playback_position()

        if pos is not None:
            text, duration, elapsed = pos
            if text and duration > 0:
                progress = max(0.0, min(1.0, elapsed / duration))
                char_idx = int(progress * len(text))
                is_open = char_idx % 2 == 0
                expr = "talk" if is_open else "normal"
            else:
                self._talk_frame = not self._talk_frame
                expr = "talk" if self._talk_frame else "normal"

            if not self._talking:
                self._talking = True
            self._apply_pixmap(expr)

        elif self._talking:
            self._talking = False
            self._apply_pixmap(self._current_expression)

    # ================================================================
    # 情绪衰减
    # ================================================================
    def _check_emotion_decay(self):
        if self._state.check_emotion_decay(EMOTION_DECAY_TIMEOUT_SEC):
            self._state.refresh_system_prompt()
            self._show_expression("normal")
            print(f"[情绪衰减] → normal")

    # ================================================================
    # 主动聊天检查
    # ================================================================
    def _check_active_chat(self):
        if self._active_chat_cooldown:
            return

        idle_time = time.time() - self._worker.last_user_input_time

        if idle_time >= ACTIVE_CHAT_IDLE_SEC:
            self._active_chat_cooldown = True
            self._worker._active_chat_queue.put(True)
            QTimer.singleShot(60_000, self._clear_active_chat_cooldown)

    def _clear_active_chat_cooldown(self):
        self._active_chat_cooldown = False

    # ================================================================
    # 信号处理
    # ================================================================
    def on_status(self, text):
        self.status_label.setText(f"状态：{text}")

    def on_user_msg(self, text):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertHtml(
            f'<div style="text-align:right;margin:4px 0;">'
            f'<span style="background-color:#4a6fff;color:white;'
            f'padding:6px 12px;border-radius:10px;display:inline-block;">'
            f'{text}</span></div>'
        )
        self._scroll_to_bottom()

    def on_catgirl_token(self, token):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertHtml(
            f'<div style="text-align:left;margin:4px 0;">'
            f'<span style="background-color:#ff6b9d;color:white;'
            f'padding:6px 12px;border-radius:10px;display:inline-block;">'
            f'{token}</span></div>'
        )
        self._scroll_to_bottom()

    def on_catgirl_done(self, full_text):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText("\n")

    def on_request_close(self):
        QTimer.singleShot(1500, self.close)

    def _update_favor_ui(self, value):
        self.favor_label.setText(f"❤️ 好感度  {value} / 100")
        self.favor_bar.setValue(value)
        self.favor_bar.setFormat(f"{value} / 100")

    def _on_favor_change(self, change):
        if change == 0:
            return
        sign = "+" if change > 0 else ""
        current = self.favor_bar.value()
        self.favor_bar.setFormat(f"{current} / 100 ({sign}{change})")
        QTimer.singleShot(2000, lambda: self.favor_bar.setFormat(f"{self.favor_bar.value()} / 100"))

    def _on_emotion_change(self, emotion):
        self._state.refresh_system_prompt()
        self._show_expression(emotion)

    def _scroll_to_bottom(self):
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ================================================================
    # 窗口事件
    # ================================================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._talking:
            self._apply_pixmap("talk" if self._talk_frame else "normal")
        elif hasattr(self, '_current_expression'):
            self._apply_pixmap(self._current_expression)

    def closeEvent(self, event):
        self._worker.requestInterruption()
        self._worker.wait(3000)
        self._tts.shutdown()
        self._state.save_all()
        event.accept()
