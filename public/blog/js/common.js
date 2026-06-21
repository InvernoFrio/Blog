/**
 * common.js - 博客公共功能模块
 * 使用方法：<script src="js/common.js"></script> 放在 </body> 前
 *
 * 功能：
 *   1. 主题切换 (dark/light)
 *   2. 移动端菜单
 *   3. 导航栏滚动阴影
 *   4. 打字机效果（通过 data-title 属性传入标题）
 *   5. 代码块复制按钮
 *   6. 图片 Lightbox
 *   7. 文章标题锚点
 *   8. 阅读时间计算
 *   9. 图片懒加载
 *  10. 返回顶部按钮
 */
(function () {
  'use strict';

  // =============================================
  // 1. 主题切换
  // =============================================
  (function initTheme() {
    var toggle = document.getElementById('themeToggle');
    if (!toggle) return;
    var root = document.documentElement;

    var saved = localStorage.getItem('theme') || 'light';
    root.setAttribute('data-theme', saved);
    toggle.textContent = saved === 'dark' ? '☀️' : '🌙';

    toggle.addEventListener('click', function () {
      var current = root.getAttribute('data-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      toggle.textContent = next === 'dark' ? '☀️' : '🌙';
    });
  })();

  // =============================================
  // 2. 移动端菜单
  // =============================================
  (function initMobileMenu() {
    var btn = document.getElementById('menuToggle');
    var links = document.getElementById('navLinks');
    if (!btn || !links) return;

    btn.addEventListener('click', function () {
      links.classList.toggle('show');
    });
  })();

  // =============================================
  // 3. 导航栏滚动阴影
  // =============================================
  (function initNavbarScroll() {
    var navbar = document.getElementById('navbar');
    if (!navbar) return;

    window.addEventListener('scroll', function () {
      if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    }, { passive: true });
  })();

  // =============================================
  // 4. 打字机效果
  // =============================================
  (function initTypewriter() {
    var el = document.getElementById('pageTitle') || document.getElementById('coverTitle');
    if (!el) return;

    // 优先从 data-title 读取，其次取 .cover-subtitle 的文本
    var text = el.getAttribute('data-title');
    if (!text) {
      var subtitle = el.parentElement && el.parentElement.querySelector('.cover-subtitle');
      if (subtitle) text = subtitle.textContent.trim();
    }
    if (!text) return;

    var i = 0;
    function type() {
      if (i < text.length) {
        i++;
        el.innerHTML = text.substring(0, i) + '<span class="typing-cursor"></span>';
        setTimeout(type, 150);
      } else {
        el.innerHTML = text;
      }
    }

    window.addEventListener('load', function () {
      setTimeout(type, 500);
    });
  })();

  // =============================================
  // 6. 代码块复制按钮
  // =============================================
  (function initCodeCopy() {
    var codeBlocks = document.querySelectorAll('pre > code');
    if (!codeBlocks.length) return;

    codeBlocks.forEach(function (code) {
      var pre = code.parentElement;
      if (pre.parentElement && pre.parentElement.classList.contains('code-wrapper')) return;

      var wrapper = document.createElement('div');
      wrapper.className = 'code-wrapper';
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      var btn = document.createElement('button');
      btn.className = 'code-copy-btn';
      btn.textContent = '复制';
      wrapper.appendChild(btn);

      btn.addEventListener('click', function () {
        var text = code.textContent;
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = '已复制!';
          setTimeout(function () { btn.textContent = '复制'; }, 2000);
        }).catch(function () {
          // fallback
          var ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          btn.textContent = '已复制!';
          setTimeout(function () { btn.textContent = '复制'; }, 2000);
        });
      });
    });
  })();

  // =============================================
  // 7. 图片 Lightbox
  // =============================================
  (function initLightbox() {
    var body = document.querySelector('.article-body');
    if (!body) return;

    var overlay = document.createElement('div');
    overlay.className = 'lightbox-overlay';
    var img = document.createElement('img');
    overlay.appendChild(img);
    document.body.appendChild(overlay);

    overlay.addEventListener('click', function () {
      overlay.classList.remove('active');
      setTimeout(function () { overlay.style.display = 'none'; }, 300);
    });

    body.querySelectorAll('img').forEach(function (image) {
      image.style.cursor = 'zoom-in';
      image.addEventListener('click', function () {
        img.src = image.src;
        img.alt = image.alt || '';
        overlay.style.display = 'flex';
        // 强制重绘以触发 transition
        overlay.offsetHeight;
        overlay.classList.add('active');
      });
    });
  })();

  // =============================================
  // 8. 文章标题锚点
  // =============================================
  (function initHeadingAnchors() {
    var body = document.querySelector('.article-body');
    if (!body) return;

    body.querySelectorAll('h2, h3').forEach(function (heading) {
      if (!heading.id) {
        heading.id = heading.textContent.trim()
          .toLowerCase()
          .replace(/[^\w\u4e00-\u9fff]+/g, '-')
          .replace(/^-|-$/g, '');
      }
      var anchor = document.createElement('a');
      anchor.className = 'heading-anchor';
      anchor.href = '#' + heading.id;
      anchor.textContent = '#';
      anchor.setAttribute('aria-label', '锚点链接');
      heading.appendChild(anchor);
    });
  })();

  // =============================================
  // 9. 阅读时间计算
  // =============================================
  (function initReadingTime() {
    var body = document.querySelector('.article-body');
    if (!body) return;

    var text = body.textContent || '';
    // 去掉空白后按中文字数 + 英文词数估算
    var chineseChars = (text.match(/[\u4e00-\u9fff]/g) || []).length;
    var englishWords = text.replace(/[\u4e00-\u9fff]/g, ' ').split(/\s+/).filter(Boolean).length;
    var minutes = Math.max(1, Math.ceil((chineseChars + englishWords * 2) / 400));

    var metaDate = document.querySelector('.meta-date');
    if (metaDate) {
      var span = document.createElement('span');
      span.className = 'reading-time';
      span.textContent = '⏱ ' + minutes + ' 分钟阅读';
      metaDate.parentNode.insertBefore(span, metaDate.nextSibling);
    }
  })();

  // =============================================
  // 10. 图片懒加载
  // =============================================
  (function initLazyLoad() {
    document.querySelectorAll('img').forEach(function (img) {
      // 跳过封面图和已经设置过的
      if (img.closest('.cover') || img.closest('.cover-hero')) return;
      if (img.hasAttribute('loading')) return;
      img.setAttribute('loading', 'lazy');
    });
  })();

  // =============================================
  // 11. 返回顶部按钮
  // =============================================
  (function initBackToTop() {
    var btn = document.createElement('button');
    btn.id = 'backToTop';
    btn.className = 'back-to-top';
    btn.innerHTML = '↑';
    btn.setAttribute('aria-label', '返回顶部');
    document.body.appendChild(btn);

    var visible = false;
    function updateVisibility() {
      var scrollY = window.scrollY || window.pageYOffset;
      if (scrollY > 300 && !visible) {
        btn.classList.add('show');
        visible = true;
      } else if (scrollY <= 300 && visible) {
        btn.classList.remove('show');
        visible = false;
      }
    }

    window.addEventListener('scroll', function () {
      updateVisibility();
    }, { passive: true });

    btn.addEventListener('click', function () {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });

    // 初始检查
    updateVisibility();
  })();

})();
