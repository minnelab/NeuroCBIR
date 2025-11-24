# neurocbir/__main__.py
from neurocbir.main import main
from neurocbir.dependencies import check_heavy_dependencies

check_heavy_dependencies()

if __name__ == "__main__":
    main()


