@behavior
class SearchCube(ActorBT):
    desc = 'search cube until confidence reaches threshold'

    def __init__(self, name, node, threshold=0.80):
        # system.pyのnavigationサブシステムを指定
        super(SearchCube, self).__init__(
            name,
            'navigation'
        )

        self.threshold = float(threshold)

    def initialise(self):
        # actorを使う準備
        super().prepare()

        # navigation内のsearch_cube actorを指定
        self.shared.set_callee([
            (
                'search_cube',
                (self.threshold,)
            )
        ])

        # actorを実行
        self.run()