// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="index.html"><strong aria-hidden="true">1.</strong> Introduction</a></li><li class="chapter-item expanded "><a href="features.html"><strong aria-hidden="true">2.</strong> Supported Features</a></li><li class="chapter-item expanded "><a href="faq.html"><strong aria-hidden="true">3.</strong> Frequently Asked Questions</a></li><li class="chapter-item expanded "><a href="guide/index.html"><strong aria-hidden="true">4.</strong> Guide</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="guide/getting_started.html"><strong aria-hidden="true">4.1.</strong> Getting Started</a></li><li class="chapter-item expanded "><a href="guide/tips.html"><strong aria-hidden="true">4.2.</strong> Tips</a></li><li class="chapter-item expanded "><a href="guide/kernel_abi.html"><strong aria-hidden="true">4.3.</strong> Kernel ABI</a></li><li class="chapter-item expanded "><a href="guide/safety.html"><strong aria-hidden="true">4.4.</strong> Safety</a></li></ol></li><li class="chapter-item expanded "><a href="cuda/index.html"><strong aria-hidden="true">5.</strong> The CUDA Toolkit</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="cuda/gpu_computing.html"><strong aria-hidden="true">5.1.</strong> GPU Computing</a></li><li class="chapter-item expanded "><a href="cuda/pipeline.html"><strong aria-hidden="true">5.2.</strong> The CUDA Pipeline</a></li></ol></li><li class="chapter-item expanded "><a href="nvvm/index.html"><strong aria-hidden="true">6.</strong> rustc_codegen_nvvm</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="nvvm/technical/index.html"><strong aria-hidden="true">6.1.</strong> Technical</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="nvvm/technical/backends.html"><strong aria-hidden="true">6.1.1.</strong> Custom Rustc Backends</a></li><li class="chapter-item expanded "><a href="nvvm/technical/nvvm.html"><strong aria-hidden="true">6.1.2.</strong> rustc_codegen_nvvm</a></li><li class="chapter-item expanded "><a href="nvvm/technical/types.html"><strong aria-hidden="true">6.1.3.</strong> Types</a></li></ol></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
