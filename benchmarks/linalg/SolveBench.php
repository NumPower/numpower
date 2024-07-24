<?php
    class SolveBench
    {
        /**
        * @var testArrayA
        * @var testArrayB
        */
        private $testArrayA = [];

        private $testArrayB = [];

        public function setUp(array $params): void
        {
            $this->testArrayA = $params['testArrayA'];
            $this->testArrayB = $params['testArrayB'];
        }

        /**
        * @BeforeMethods("setUp")
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideArrays"
        * })
        */
        public function benchSolve($params): void
        {
            \NDArray::solve($this->testArrayA, $this->testArrayB);
        }

        public function provideArrays() {
            yield ['testArrayA' => \NDArray::ones([2, 2]), 'testArrayB' => \NDArray::full([2, 2], 10)];
            yield ['testArrayA' => \NDArray::ones([10, 10]), 'testArrayB' => \NDArray::full([10, 10], 10)];
            yield ['testArrayA' => \NDArray::ones([500, 500]), 'testArrayB' => \NDArray::full([500, 500], 10)];
            yield ['testArrayA' => \NDArray::ones([1000, 1000]), 'testArrayB' => \NDArray::full([1000, 1000], 10)];
        }
    }
?>
