'use client';

import Link from 'next/link';
import { useState, useRef } from 'react';

export function Navbar() {
  const [open, setOpen] = useState(false);
  const [topicOpen, setTopicOpen] = useState(false);

  // 用 ref 追踪定时器，避免状态更新导致的竞态
  const openTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const topicOpenTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearOpenTimer = () => {
    if (openTimer.current) {
      clearTimeout(openTimer.current);
      openTimer.current = null;
    }
  };

  const clearTopicOpenTimer = () => {
    if (topicOpenTimer.current) {
      clearTimeout(topicOpenTimer.current);
      topicOpenTimer.current = null;
    }
  };

  return (
    <nav
      style={{
        position: 'fixed',
        top: '16px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 1000,
        width: 'fit-content',
        maxWidth: '90vw',
        borderRadius: '50px',
        overflow: 'visible',
      }}
    >
      {/* 主胶囊容器 */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.05) 100%)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRadius: '50px',
          padding: '8px 8px 8px 20px',
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          boxShadow: `
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.2),
            inset 0 -1px 0 rgba(0,0,0,0.1),
            0 0 60px rgba(0,200,255,0.15),
            0 0 40px rgba(255,200,100,0.1)
          `,
          border: '1px solid rgba(255,255,255,0.1)',
          position: 'relative',
          overflow: 'visible',
        }}
      >
        {/* 顶部边缘高光 */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: '10%',
            right: '10%',
            height: '1px',
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent)',
          }}
        />

        {/* 左侧图标组 */}
        <div className="flex items-center gap-1">
          {[
            { href: '/', icon: 'fa-house', label: '首页' },
            { href: '/about', icon: 'fa-user', label: '关于' },
          ].map((item) => (
            <Link key={item.href} href={item.href} className="nav-link">
              <i className={`fa-solid ${item.icon}`} style={{ fontSize: '12px' }}></i>
              <span>{item.label}</span>
            </Link>
          ))}
        </div>

        {/* 垂直分隔线 */}
        <div
          style={{
            width: '1px',
            height: '24px',
            background: 'linear-gradient(180deg, transparent 0%, rgba(0,200,255,0.5) 30%, rgba(255,200,100,0.5) 70%, transparent 100%)',
            boxShadow: '0 0 8px rgba(0,200,255,0.3)',
          }}
        />

        {/* 右侧图标组 */}
        <div className="flex items-center gap-1">
          {/* 博文下拉 */}
          <div
            style={{ position: 'relative' }}
            onMouseEnter={() => {
              clearOpenTimer();
              setOpen(true);
            }}
            onMouseLeave={() => {
              clearOpenTimer();
              openTimer.current = setTimeout(() => setOpen(false), 200);
            }}
          >
            <div
              className="nav-link"
              style={{
                background: open ? 'rgba(255,255,255,0.2)' : undefined,
              }}
              onClick={() => setOpen(!open)}
            >
              <i className="fa-solid fa-book-open" style={{ fontSize: '12px' }}></i>
              <span>博文</span>
              <i
                className="fa-solid fa-chevron-down"
                style={{
                  fontSize: '10px',
                  transition: 'transform 0.2s',
                  transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
                }}
              ></i>
            </div>

            {open && (
              <div
                style={{
                  position: 'absolute',
                  top: 'calc(100% + 8px)',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  minWidth: '160px',
                  background: 'rgba(20, 20, 30, 0.95)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255,255,255,0.15)',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                  overflow: 'hidden',
                  zIndex: 2000,
                }}
              >
                <a href="/posts/diffusion-index" className="nav-dropdown-item">
                  <i className="fa-solid fa-wand-magic-sparkles" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  Diffusion 系列
                </a>
                <a href="/posts/vla-index" className="nav-dropdown-item">
                  <i className="fa-solid fa-robot" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  VLA 系列
                </a>
                <a href="/posts/vla-learning-index" className="nav-dropdown-item">
                  <i className="fa-solid fa-graduation-cap" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  VLA 入门系列
                </a>
                <a href="/archives" className="nav-dropdown-item">
                  <i className="fa-solid fa-list-ul" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  全部博文
                </a>
              </div>
            )}
          </div>

          {/* 专题下拉 */}
          <div
            style={{ position: 'relative' }}
            onMouseEnter={() => {
              clearTopicOpenTimer();
              setTopicOpen(true);
            }}
            onMouseLeave={() => {
              clearTopicOpenTimer();
              topicOpenTimer.current = setTimeout(() => setTopicOpen(false), 200);
            }}
          >
            <div
              className="nav-link"
              style={{
                background: topicOpen ? 'rgba(255,255,255,0.2)' : undefined,
              }}
              onClick={() => setTopicOpen(!topicOpen)}
            >
              <i className="fa-solid fa-layer-group" style={{ fontSize: '12px' }}></i>
              <span>专题</span>
              <i
                className="fa-solid fa-chevron-down"
                style={{
                  fontSize: '10px',
                  transition: 'transform 0.2s',
                  transform: topicOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                }}
              ></i>
            </div>

            {topicOpen && (
              <div
                style={{
                  position: 'absolute',
                  top: 'calc(100% + 8px)',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  minWidth: '200px',
                  background: 'rgba(20, 20, 30, 0.95)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255,255,255,0.15)',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                  overflow: 'hidden',
                  zIndex: 2000,
                }}
              >
                <a href="/topics/ad-prediction-planning" className="nav-dropdown-item">
                  <i className="fa-solid fa-car" style={{ fontSize: '12px', color: 'rgba(59,130,246,0.8)' }}></i>
                  自动驾驶预测与规划
                </a>
                <a href="/topics/diffusion" className="nav-dropdown-item">
                  <i className="fa-solid fa-wand-magic-sparkles" style={{ fontSize: '12px', color: 'rgba(34,197,94,0.8)' }}></i>
                  Diffusion 模型
                </a>
                <a href="/topics/rl" className="nav-dropdown-item">
                  <i className="fa-solid fa-brain" style={{ fontSize: '12px', color: 'rgba(245,158,11,0.8)' }}></i>
                  强化学习
                </a>
                <a href="/topics/vla" className="nav-dropdown-item">
                  <i className="fa-solid fa-robot" style={{ fontSize: '12px', color: 'rgba(124,58,237,0.8)' }}></i>
                  VLA 模型
                </a>
              </div>
            )}
          </div>

          {[
            { href: '/tags', icon: 'fa-tag', label: '标签' },
          ].map((item) => (
            <Link key={item.href} href={item.href} className="nav-link">
              <i className={`fa-solid ${item.icon}`} style={{ fontSize: '12px' }}></i>
              <span>{item.label}</span>
            </Link>
          ))}
        </div>

        {/* Logo 文字 */}
        <Link href="/" className="nav-logo">
          YoRHa::LaB
        </Link>

        {/* 用户头像插槽 */}
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.05) 100%)',
            border: '1px solid rgba(255,255,255,0.2)',
            boxShadow: `
              0 4px 16px rgba(0,0,0,0.2),
              inset 0 1px 0 rgba(255,255,255,0.3),
              0 0 20px rgba(0,200,255,0.2)
            `,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            marginLeft: '8px',
          }}
        >
          <i className="fa-solid fa-circle-user" style={{ color: 'rgba(30,30,30,0.6)', fontSize: '16px' }}></i>
        </div>
      </div>
    </nav>
  );
}