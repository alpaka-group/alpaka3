(function () {
    const originalScrollIntoView = Element.prototype.scrollIntoView;

    Element.prototype.scrollIntoView = function (...args) {
        if (this.closest && this.closest(".wy-menu-vertical")) {
            return;
        }
        return originalScrollIntoView.apply(this, args);
    };

    function keepActiveTopicVisible() {
        const sidebar = document.querySelector(".wy-side-scroll");
        if (!sidebar) {
            return;
        }

        const activeItem = document.querySelector(
            ".wy-menu-vertical li.current, .wy-menu-vertical a.current"
        );
        if (!activeItem) {
            return;
        }

        const sidebarRect = sidebar.getBoundingClientRect();
        const itemRect = activeItem.getBoundingClientRect();
        const itemIsVisible =
            itemRect.top >= sidebarRect.top && itemRect.bottom <= sidebarRect.bottom;

        if (itemIsVisible) {
            return;
        }

        const itemOffset = activeItem.offsetTop - sidebar.offsetTop;
        const targetScrollTop = itemOffset - sidebar.clientHeight / 2 + activeItem.clientHeight / 2;
        sidebar.scrollTop = Math.max(0, targetScrollTop);
    }

    document.addEventListener("DOMContentLoaded", keepActiveTopicVisible);
    window.addEventListener("load", function () {
        keepActiveTopicVisible();
        window.setTimeout(keepActiveTopicVisible, 0);
        window.setTimeout(keepActiveTopicVisible, 100);
    });
})();
