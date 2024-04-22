from ._base import App


class Example(App):
    """Example application to test the API."""

    def add_custom_cli_args(self):
        self.parser.add_argument("--foo", action="store_true", help="Should I foo?")
        self.parser.add_argument("--num", type=int, default=1, help="How many times?")

    def validate_and_transform_args(self, args):
        assert args.num < 5, "NUM must be less than 5"
        return args

    def execute(self, args):
        if args.foo:
            for i in range(args.num):
                print("Foo!")
        else:
            print("Nope, that's a bar.")
