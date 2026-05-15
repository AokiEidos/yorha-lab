'use client';

import Link from 'next/link';

const topics = [
  {
    id: 'ad-prediction-planning',
    title: '自动驾驶预测与规划',
    subtitle: 'Autonomous Driving Prediction & Planning',
    description: '覆盖自动驾驶预测与规划从感知到端到端全栈技术脉络',
    icon: 'fa-car',
    color: '#3b82f6',
    chapters: 12,
    years: '2024–2026',
  },
  {
    id: 'diffusion',
    title: 'Diffusion Model',
    subtitle: '扩散模型技术文档',
    description: '从扩散过程基础到多模态生成与自动驾驶应用的完整技术体系',
    icon: 'fa-wand-magic-sparkles',
    color: '#22c55e',
    chapters: 12,
    years: '2023–2026',
  },
  {
    id: 'rl',
    title: 'Reinforcement Learning',
    subtitle: '强化学习技术文档',
    description: '从 MDP 基础到 RLHF 与多智能体系统的完整技术脉络',
    icon: 'fa-brain',
    color: '#f59e0b',
    chapters: 12,
    years: '2023–2026',
  },
  {
    id: 'vla',
    title: 'VLA: Vision-Language-Action',
    subtitle: '视觉-语言-动作模型技术文档',
    description: '从 VLA 基础到机器人操控与端到端驾驶的前沿技术体系',
    icon: 'fa-robot',
    color: '#7c3aed',
    chapters: 12,
    years: '2024–2026',
  },
];

export default function TopicsPage() {
  return (
    <div
      style={{
        minHeight: '100vh',
        padding: '120px 24px 60px',
        maxWidth: '1100px',
        margin: '0 auto',
      }}
    >
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '48px' }}>
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '16px',
          }}
        >
          <i
            className="fa-solid fa-layer-group"
            style={{ fontSize: '28px', color: '#3b82f6' }}
          />
          <h1
            style={{
              fontSize: '32px',
              fontWeight: 700,
              color: '#1e293b',
              letterSpacing: '-0.5px',
            }}
          >
            技术专题
          </h1>
        </div>
        <p style={{ color: '#64748b', fontSize: '14px' }}>
          深入探索核心技术领域的完整知识体系
        </p>
      </div>

      {/* Topic Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: '20px',
        }}
      >
        {topics.map((topic) => (
          <Link
            key={topic.id}
            href={`/topics/${topic.id}`}
            style={{
              display: 'block',
              background: '#fff',
              borderRadius: '12px',
              padding: '24px',
              border: '1px solid #e2e8f0',
              borderTop: `3px solid ${topic.color}`,
              textDecoration: 'none',
              transition: 'transform 0.15s, box-shadow 0.15s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-3px)';
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
              <div
                style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: '10px',
                  background: `${topic.color}15`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <i className={`fa-solid ${topic.icon}`} style={{ fontSize: '18px', color: topic.color }} />
              </div>
              <div>
                <h2 style={{ fontSize: '16px', fontWeight: 700, color: '#1e293b', marginBottom: '2px' }}>
                  {topic.title}
                </h2>
                <p style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600 }}>
                  {topic.subtitle}
                </p>
              </div>
            </div>

            <p style={{ fontSize: '13px', color: '#64748b', marginBottom: '16px', lineHeight: 1.6 }}>
              {topic.description}
            </p>

            <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#94a3b8' }}>
              <span>
                <i className="fa-solid fa-book" style={{ marginRight: '4px' }} />
                {topic.chapters} 章
              </span>
              <span>
                <i className="fa-solid fa-calendar" style={{ marginRight: '4px' }} />
                {topic.years}
              </span>
            </div>
          </Link>
        ))}
      </div>

      {/* Bottom note */}
      <div
        style={{
          marginTop: '48px',
          textAlign: 'center',
          padding: '16px',
          background: '#f8fafc',
          borderRadius: '8px',
          border: '1px solid #e2e8f0',
        }}
      >
        <p style={{ fontSize: '12px', color: '#94a3b8' }}>
          <i className="fa-solid fa-circle-info" style={{ marginRight: '6px' }} />
          点击任意专题进入文档索引页面，每个专题包含完整的章节导航与内容
        </p>
      </div>
    </div>
  );
}
