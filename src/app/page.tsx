'use client';

import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import Link from 'next/link';

interface PostMeta {
  slug: string;
  title: string;
  date: string;
  tags: string[];
  excerpt: string;
  views?: number;
  readingTime?: number;
  isPinned?: boolean;
  words?: number;
}

interface Stats {
  days: number;
  views: number;
  lastOnline: string;
  totalPosts: number;
  totalWords: number;
}

export default function Home() {
  const [posts, setPosts] = useState<PostMeta[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [stats, setStats] = useState<Stats>({ days: 0, views: 0, lastOnline: '刚刚', totalPosts: 0, totalWords: 0 });
  const [loading, setLoading] = useState(true);
  const [bgImage, setBgImage] = useState('/images/nier.jpg');

  useEffect(() => {
    fetch('/api/posts')
      .then(res => res.json())
      .then(data => {
        setPosts(data.posts || []);
        const allTags = (data.tags || []);
        // 过滤：只显示出现≥2次的标签，减少噪音
        const tagCount: Record<string, number> = {};
        (data.posts || []).forEach((post: any) => {
          (post.tags || []).forEach((t: string) => { tagCount[t] = (tagCount[t] || 0) + 1; });
        });
        const filtered = allTags.filter((t: string) => (tagCount[t] || 0) >= 2);
        setTags(filtered);
        setStats(data.stats || { days: 0, views: 0, lastOnline: '刚刚', totalPosts: 0, totalWords: 0 });
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
    
    setBgImage('/images/nier.jpg');
  }, []);

  return (
    <div className="min-h-screen">
      {/* 底部白色背景层 */}
      <div className="bg-bottom"></div>

      {/* Hero 区 - Sticky 定位 */}
      <div
        className="hero-section"
        style={{ backgroundImage: `url(${bgImage})` }}
      >
        {/* 顶部占位 - 为固定导航栏留空间 */}
        <div className="h-14"></div>
        {/* 头像区域 - 位于首页上方 1/3 */}
        <div className="relative h-screen flex flex-col items-center justify-start" style={{ paddingTop: '12vh' }}>
          {/* 头像 */}
          <div className="w-24 h-24 sm:w-32 sm:h-32 rounded-full overflow-hidden border-4 border-white/30 shadow-lg mb-4">
            <img 
              src="/images/shiki.jpeg" 
              alt="avatar"
              className="w-full h-full object-cover"
            />
          </div>
          
          {/* 标题 */}
          <h1 className="text-2xl sm:text-3xl font-bold text-white mb-2" style={{ fontFamily: 'Zen Antique Soft, serif', textShadow: '0 2px 10px rgba(0,0,0,0.3)' }}>
            YoRHa::LaB
          </h1>
          
          {/* 描述 */}
          <p className="text-white/80 text-sm" style={{ textShadow: '0 1px 5px rgba(0,0,0,0.3)' }}>
            YOU CAN (NOT) ADVANCE
          </p>
        </div>

        {/* 统计栏 */}
        <div className="absolute bottom-0 left-0 right-0 border-b border-gray-200/30">
          <div className="container mx-auto px-4 py-3">
            <div className="flex flex-wrap justify-center gap-6 sm:gap-10 text-center">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{stats.days}</div>
                <div className="text-xs text-gray-400">天</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{stats.totalPosts}</div>
                <div className="text-xs text-gray-400">文章</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">
                  {stats.totalWords > 0 ? `${Math.round(stats.totalWords / 1000)}k` : '0'}
                </div>
                <div className="text-xs text-gray-400">字</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-white">{stats.lastOnline}</div>
                <div className="text-xs text-gray-400">在线</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Content 区 - 毛玻璃效果 */}
      <div className="content-section">
        {/* 分隔线渐变阴影 */}
        <div className="divider-gradient"></div>
        
        {/* 主内容区 */}
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            {/* 文章列表标题 */}
            <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-gray-300" style={{ fontFamily: 'Kaisei Decol, serif' }}>
              <i className="fa-solid fa-book-open"></i>
              <span>文章列表</span>
            </h2>
          
          {loading ? (
            <div className="space-y-4">
              {[1, 2].map(i => (
                <div key={i} className="card animate-pulse">
                  <div className="flex">
                    <div className="w-48 h-32 bg-gray-200 rounded-lg shrink-0"></div>
                    <div className="ml-4 flex-1">
                      <div className="h-6 bg-gray-200 rounded mb-2 w-3/4"></div>
                      <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : posts.length === 0 ? (
            <div className="card">
              <p className="text-muted">暂无论文笔记</p>
            </div>
          ) : (
            <div className="space-y-4">
              {posts.map((post, index) => (
                <article 
                  key={post.slug}
                  className={`card cursor-pointer group ${post.isPinned ? 'pinned' : ''}`}
                  style={{ 
                    animationDelay: `${index * 0.1}s`,
                    '--article-highlight': 'rgba(122, 162, 247, 0.2)'
                  } as React.CSSProperties}
                >
                  <Link href={`/posts/${post.slug}`} className="flex flex-col sm:flex-row gap-4">
                    {/* 封面图 - 左侧 */}
                    <div className="card-image relative w-full sm:w-48 h-32 shrink-0 overflow-hidden rounded-lg bg-gradient-to-br from-orange-100 to-purple-100 dark:from-orange-900/30 dark:to-purple-900/30">
                      <div className="w-full h-full flex items-center justify-center">
                        <i className="fa-solid fa-file-lines text-3xl text-gray-400"></i>
                      </div>
                      {/* 置顶角标 */}
                      {post.isPinned && (
                        <span className="absolute top-2 left-2 px-2 py-0.5 bg-orange-500 text-white text-xs rounded">
                          <i className="fa-solid fa-chess-queen mr-1"></i>置顶
                        </span>
                      )}
                    </div>
                    
                    {/* 内容区域 */}
                    <div className="flex-1 min-w-0">
                      {/* 日期 */}
                      <div className="text-xs text-gray-500 mb-1">
                        <i className="fa-regular fa-clock mr-1"></i>
                        发布于 {format(new Date(post.date), 'yyyy-MM-dd HH:mm:ss', { locale: zhCN })}
                      </div>
                      
                      {/* 标题 */}
                      <h3 className="text-lg font-semibold mb-2 group-hover:text-blue-400 transition-colors line-clamp-1" style={{ fontFamily: 'Kaisei Decol, serif' }}>
                        {post.title}
                      </h3>
                      
                      {/* AI摘要 */}
                      <p className="text-sm text-gray-400 mb-2 line-clamp-2">
                        {post.excerpt}
                      </p>
                      
                      {/* 元信息 */}
                      <div className="flex items-center gap-3 text-xs text-gray-500">
                        {post.words && post.words > 0 && (
                          <span>
                            <i className="fa-solid fa-pen-nib mr-1"></i>
                            {post.words.toLocaleString()} 字
                          </span>
                        )}
                        {post.readingTime && (
                          <span>
                            <i className="fa-regular fa-clock mr-1"></i>
                            {post.readingTime} 分钟
                          </span>
                        )}
                      </div>
                    </div>
                  </Link>
                </article>
              ))}
            </div>
          )}

          {/* 分页提示 */}
          {!loading && posts.length > 0 && (
            <div className="mt-8 text-center">
              <span className="text-sm text-gray-500">- 已加载全部 {posts.length} 篇 -</span>
            </div>
          )}
          
          {/* 标签 */}
          {tags.length > 0 && (
            <div className="mt-8">
              <h3 className="text-sm font-semibold mb-3 text-gray-400">
                <i className="fa-solid fa-tags mr-2"></i>
                标签
              </h3>
              <div className="flex flex-wrap gap-2">
                {tags.map((tag) => (
                  <Link
                    key={tag}
                    href={`/tags/${tag}`}
                    className="tag hover:no-underline"
                  >
                    {tag}
                  </Link>
                ))}
              </div>
            </div>
          )}

          {/* 底部留白 - 确保页面底部是白色 */}
          <div className="h-32"></div>
          </div>
        </div>
      </div>
    </div>
  );
}
