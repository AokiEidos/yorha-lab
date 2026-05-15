import { notFound } from 'next/navigation';
import type { Metadata } from 'next';

interface PageProps {
  params: Promise<{ topic: string }>;
}

const validTopics = ['ad-prediction-planning', 'diffusion', 'rl', 'vla'];

const topicMeta: Record<string, { title: string; description: string }> = {
  'ad-prediction-planning': {
    title: '自动驾驶预测与规划',
    description: '自动驾驶预测与规划技术文档系列',
  },
  diffusion: {
    title: 'Diffusion Model',
    description: '扩散模型技术文档系列',
  },
  rl: {
    title: 'Reinforcement Learning',
    description: '强化学习技术文档系列',
  },
  vla: {
    title: 'VLA: Vision-Language-Action',
    description: '视觉-语言-动作模型技术文档系列',
  },
};

export async function generateStaticParams() {
  return validTopics.map((topic) => ({ topic }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { topic } = await params;
  const meta = topicMeta[topic];
  if (!meta) return { title: '未找到' };
  return {
    title: `${meta.title} — YoRHa::LaB`,
    description: meta.description,
  };
}

export default async function TopicPage({ params }: PageProps) {
  const { topic } = await params;

  if (!validTopics.includes(topic)) {
    notFound();
  }

  const meta = topicMeta[topic];

  return (
    <div style={{ minHeight: '100vh', paddingTop: '80px' }}>
      {/* 返回导航 */}
      <div style={{
        padding: '16px 32px 0',
      }}>
        <a href="/topics" style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          color: '#64748b',
          fontSize: '13px',
          textDecoration: 'none',
        }}>
          <i className="fa-solid fa-arrow-left" />
          返回专题总览
        </a>
      </div>

      {/* Iframe 承载 HTML 内容 — 全宽无边距 */}
      <div style={{
        padding: '16px 32px 60px',
      }}>
        <div style={{
          background: '#fff',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          overflow: 'hidden',
          minHeight: '90vh',
        }}>
          <iframe
            src={`/topics/${topic}/index.html`}
            style={{
              width: '100%',
              minHeight: '90vh',
              border: 'none',
            }}
            title={meta.title}
          />
        </div>
      </div>
    </div>
  );
}