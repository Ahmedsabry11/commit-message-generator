GIT_DIFF = """
diff --git a/app.py b/app.py
index 123abc..456def 100644
--- a/app.py
+++ b/app.py
@@ def calculate_sum(a, b):
-    return a + b
+    result = a + b
+    print(f"Sum is {result}")
+    return result
"""
