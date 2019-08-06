import nox

@nox.session(python="3.6")
def test(session):
  session.install("-e", "../../")
  # session.run("mammo2risk", "w", "../downloads/weights/", "o", ".")
  session.run("mammo2risk", "--f", "CCL_S13_425_20130625.dcm")
  # session.run("mammo2risk", "--o", ".", "--d", "../docs/samples/")