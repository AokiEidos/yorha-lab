import Link from 'next/link';
import { getAllTags, getAllPosts } from '@/lib/posts';

export default function TagsPage() {
  const tags = getAllTags();
  const posts = getAllPosts();

  // 计算每个标签的文章数
  const tagCounts = tags.reduce((acc, tag) => {
    acc[tag] = posts.filter((post) => post.tags.includes(tag)).length;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="container py-12">
      <h1 className="text-4xl font-bold mb-8">标签</h1>

      {tags.length === 0 ? (
        <div className="card">
          <p className="text-secondary">暂无标签...</p>
        </div>
      ) : (
        <div className="flex flex-wrap gap-3">
          {tags.map((tag) => (
            <Link
              key={tag}
              href={`/tags/${tag}`}
              className="px-4 py-2 bg-muted rounded-lg hover:bg-primary hover:text-white transition-colors"
            >
              {tag}
              <span className="ml-1 text-sm opacity-70">({tagCounts[tag]})</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
