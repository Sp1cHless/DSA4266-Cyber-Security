"""Language audit distinguishes prose from imported font identifiers."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from CIC_new.check_project_language import audit


class ProjectLanguageTests(unittest.TestCase):
    def test_prose_and_office_font_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"README.md").write_text("English documentation", encoding="utf-8")
            font = "\u5b8b\u4f53"
            with zipfile.ZipFile(root/"book.xlsx", "w") as archive:
                archive.writestr("xl/styles.xml", '<styleSheet><fonts><font><name val="'+font+'"/></font></fonts></styleSheet>')
            result = audit(root)
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["preserved_font_and_builtin_style_identifiers"]), 1)
            (root/"README.md").write_text("\u4e2d\u6587", encoding="utf-8")
            self.assertFalse(audit(root)["passed"])
            (root/"README.md").write_text("English", encoding="utf-8")
            with zipfile.ZipFile(root/"report.docx", "w") as archive:
                archive.writestr("word/document.xml", '<document><p><t>\u4e2d\u6587</t></p></document>')
            self.assertFalse(audit(root)["passed"])


if __name__ == "__main__":
    unittest.main()
