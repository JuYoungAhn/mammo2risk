import nox

@nox.session(python="3.6")
def test(session):
  session.install("-e", "../")
  # session.run("mammo2risk", "--o", ".")
  session.run("mammo2risk", "--o", ".", "--d", "../docs/samples/")