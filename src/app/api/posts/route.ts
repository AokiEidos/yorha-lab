import { NextResponse } from 'next/server';
import { getAllPosts, getAllTags } from '@/lib/posts';

export async function GET() {
  try {
    const posts = getAllPosts();
    const tags = getAllTags();
    
    // 计算真实统计数据
    const totalViews = posts.reduce((sum, post) => sum + (post.views || 0), 0);
    
    // 计算运行时间（从第一篇文章开始）
    let days = 1;
    if (posts.length > 0) {
      const firstPostDate = new Date(posts[posts.length - 1].date);
      const today = new Date();
      days = Math.floor((today.getTime() - firstPostDate.getTime()) / (1000 * 60 * 60 * 24)) + 1;
    }
    
    // 统计总字数
    const totalWords = posts.reduce((sum, post) => sum + (post.readingTime || 0) * 1000, 0);
    
    return NextResponse.json({
      posts,
      tags,
      stats: {
        days,
        views: totalViews,
        lastOnline: '刚刚',
        totalPosts: posts.length,
        totalWords,
      }
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load posts' }, { status: 500 });
  }
}
