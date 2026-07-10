from pytwb.common import behavior, ActorBT


@behavior
class SearchCube(ActorBT):
    desc = 'search cube until confidence reaches threshold'

    def __init__(self, name, node, threshold=0.80):
        # navigationサブシステムを指定
        super(SearchCube, self).__init__(
            name,
            'navigation'
        )

        self.threshold = float(threshold)

    def initialise(self):
        # actorを呼び出す準備
        super().prepare()

        # system.pyのsearch_cube actorを指定
        self.shared.set_callee([
            (
                'search_cube',
                (self.threshold,)
            )
        ])

        # actorを実行
        self.run()