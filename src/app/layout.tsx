import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

export const metadata: Metadata = {
  title: "YoRHa::LaB - 技术博客",
  description: "探索技术与创意的无限可能",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Kaisei+Decol:wght@400;500;700&family=Zen+Antique+Soft&display=swap" rel="stylesheet" />
      </head>
      <body className="min-h-screen">
        {/* 背景层 - 非首页显示 */}
        <div id="centerbg" className="bg-image" suppressHydrationWarning></div>
        
        <div className="site-wrapper">
          <Navbar />
          <main>
            {children}
          </main>
        </div>
        
        {/* 随机换背景按钮 */}
        <button 
          id="random-bg-btn"
          className="fixed bottom-4 right-4 z-50 w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm border border-white/30 flex items-center justify-center text-white hover:bg-white/30 transition-all cursor-pointer"
          title="随机换张背景"
        >
          <i className="fa-solid fa-shuffle"></i>
        </button>
        
        <script dangerouslySetInnerHTML={{__html: `
          // 随机选择背景图片
          (function() {
            var images = ['/images/nier.jpg'];
            var randomImg = images[Math.floor(Math.random() * images.length)];
            var bg = document.getElementById('centerbg');
            if (bg) bg.style.backgroundImage = 'url(' + randomImg + ')';
            
            // 首页隐藏全局背景（使用页面内 hero 背景）
            if (window.location.pathname === '/') {
              bg.style.display = 'none';
            }
          })();
          
          // 随机换背景按钮点击事件
          document.getElementById('random-bg-btn').addEventListener('click', function() {
            var images = ['/images/nier.jpg'];
            var currentBg = document.getElementById('centerbg').style.backgroundImage;
            var availableImages = images.filter(function(img) {
              return !currentBg.includes(img.split('/').pop());
            });
            if (availableImages.length === 0) availableImages = images;
            var randomImg = availableImages[Math.floor(Math.random() * availableImages.length)];
            document.getElementById('centerbg').style.backgroundImage = 'url(' + randomImg + ')';
          });
        `}} />
      </body>
    </html>
  );
}
