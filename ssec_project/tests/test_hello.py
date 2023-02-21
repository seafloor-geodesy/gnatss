from ssec_project import hello


def test_say_hello():
    """Tests the say_hello function"""
    hello_text = hello.say_hello()

    assert hello_text == "Hello world."
