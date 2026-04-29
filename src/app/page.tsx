import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import Link from 'next/link';
import Image from 'next/image';
import { getAllPosts, getAllTags } from '@/lib/posts';

export default function Home() {
  const posts = getAllPosts();
  const allTags = getAllTags();

  // 只显示出现≥2次的标签，减少噪音
  const tagCount: Record<string, number> = {};
  for (const post of posts) {
    for (const t of post.tags) {
      tagCount[t] = (tagCount[t] || 0) + 1;
    }
  }
  const tags = allTags.filter((t) => (tagCount[t] || 0) >= 2);

  // 统计数据
  const totalWords = posts.reduce((sum, p) => sum + (p.readingTime || 0) * 1000, 0);
  let days = 1;
  if (posts.length > 0) {
    const firstPostDate = new Date(posts[posts.length - 1].date);
    const today = new Date();
    days = Math.floor((today.getTime() - firstPostDate.getTime()) / (1000 * 60 * 60 * 24)) + 1;
  }

  return (
    <div className="min-h-screen">
      <div className="background_blur"></div>
      {/* 底部白色背景层 */}
      <div className="bg-bottom"></div>

      {/* Hero 区 - Sticky 定位 */}
      <div
        className="hero-section"
        style={{ backgroundImage: 'url(/images/nier.jpg)' }}
      >
        {/* 顶部占位 - 为固定导航栏留空间 */}
        <div className="h-14"></div>
        {/* 头像区域 - 位于首页上方 1/3 */}
        <div className="relative h-screen flex flex-col items-center justify-start" style={{ paddingTop: '12vh' }}>
          {/* 头像 */}
          <div className="w-24 h-24 sm:w-32 sm:h-32 rounded-full overflow-hidden border-4 border-white/30 shadow-lg mb-4">
            <Image
              src="/images/shiki.jpeg"
              alt="avatar"
              width={128}
              height={128}
              className="w-full h-full object-cover"
              priority
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
                <div className="text-2xl font-bold text-white">{days}</div>
                <div className="text-xs text-gray-400">天</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{posts.length}</div>
                <div className="text-xs text-gray-400">文章</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">
                  {totalWords > 0 ? `${Math.round(totalWords / 1000)}k` : '0'}
                </div>
                <div className="text-xs text-gray-400">字</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-white">刚刚</div>
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

            {posts.length === 0 ? (
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
                      <div className="card-image relative w-full sm:w-48 h-32 shrink-0 overflow-hidden rounded-lg bg-gradient-to-br from-orange-100 to-purple-100">
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
            {posts.length > 0 && (
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
