'use client';

import Link from 'next/link';
import { useState } from 'react';

export function Navbar() {
  const [open, setOpen] = useState(false);

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
            <Link
              key={item.href}
              href={item.href}
              style={{
                color: 'rgba(30,30,30,0.8)',
                padding: '8px 12px',
                borderRadius: '25px',
                fontSize: '13px',
                fontWeight: 500,
                textDecoration: 'none',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.2)';
                e.currentTarget.style.color = 'rgba(30,30,30,1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = 'rgba(30,30,30,0.8)';
              }}
            >
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
          {/* 博文下拉触发器 */}
          <div style={{ position: 'relative' }}>
            <div
              style={{
                color: 'rgba(30,30,30,0.8)',
                padding: '8px 12px',
                borderRadius: '25px',
                fontSize: '13px',
                fontWeight: 500,
                textDecoration: 'none',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                cursor: 'pointer',
                background: open ? 'rgba(255,255,255,0.2)' : 'transparent',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.2)';
                e.currentTarget.style.color = 'rgba(30,30,30,1)';
                setOpen(true);
              }}
              onMouseLeave={(e) => {
                if (!open) {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = 'rgba(30,30,30,0.8)';
                }
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

            {/* 下拉菜单 */}
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
                onMouseLeave={() => setOpen(false)}
              >
                <a
                  href="/posts/diffusion-index"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    color: 'rgba(255,255,255,0.85)',
                    textDecoration: 'none',
                    fontSize: '13px',
                    fontWeight: 500,
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <i className="fa-solid fa-wand-magic-sparkles" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  Diffusion 系列
                </a>
                <a
                  href="/posts/vla-index"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    color: 'rgba(255,255,255,0.85)',
                    textDecoration: 'none',
                    fontSize: '13px',
                    fontWeight: 500,
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <i className="fa-solid fa-robot" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  VLA 系列
                </a>
                <a
                  href="/posts/vla-learning-index"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    color: 'rgba(255,255,255,0.85)',
                    textDecoration: 'none',
                    fontSize: '13px',
                    fontWeight: 500,
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <i className="fa-solid fa-graduation-cap" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  VLA 入门系列
                </a>
                <a
                  href="/archives"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    color: 'rgba(255,255,255,0.85)',
                    textDecoration: 'none',
                    fontSize: '13px',
                    fontWeight: 500,
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <i className="fa-solid fa-list-ul" style={{ fontSize: '12px', color: 'rgba(0,200,255,0.8)' }}></i>
                  全部博文
                </a>
              </div>
            )}
          </div>

          {[
            { href: '/tags', icon: 'fa-tag', label: '标签' },
          ].map((item) => (
            <Link
              key={item.href}
              href={item.href}
              style={{
                color: 'rgba(30,30,30,0.8)',
                padding: '8px 12px',
                borderRadius: '25px',
                fontSize: '13px',
                fontWeight: 500,
                textDecoration: 'none',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.2)';
                e.currentTarget.style.color = 'rgba(30,30,30,1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = 'rgba(30,30,30,0.8)';
              }}
            >
              <i className={`fa-solid ${item.icon}`} style={{ fontSize: '12px' }}></i>
              <span>{item.label}</span>
            </Link>
          ))}
        </div>

        {/* Logo 文字 */}
        <Link
          href="/"
          style={{
            fontFamily: 'Kaisei Decol, serif',
            fontSize: '16px',
            fontWeight: 700,
            color: 'rgba(30,30,30,0.9)',
            textDecoration: 'none',
            padding: '8px 16px',
            borderRadius: '25px',
            background: 'linear-gradient(135deg, rgba(0,200,255,0.1) 0%, rgba(255,200,100,0.1) 100%)',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(0,200,255,0.25) 0%, rgba(255,200,100,0.25) 100%)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(0,200,255,0.1) 0%, rgba(255,200,100,0.1) 100%)';
          }}
        >
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
