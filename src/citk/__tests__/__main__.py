"""Allow ``python -m citk.__tests__`` to run the bundled pytest suite."""

from .run import main

raise SystemExit(main())
