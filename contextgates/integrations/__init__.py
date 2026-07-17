"""Optional framework integrations.

Nothing here is imported by ``contextgates`` itself: the core package stays
dependency-free. Import an adapter only if you already have that framework
installed.

    from contextgates.integrations.llamaindex import GateNodePostprocessor
    from contextgates.integrations.langchain import GateDocumentCompressor
"""
