<?php
    /**
     * @Groups({"linearAlgebra"})
     */
    class OuterBench
    {
        /**
        * @var testArray
        */
        private $testArray = [];

        public function setUp(array $params): void
        {
            $this->testArray = $params['testArray'];
        }

        /**
        * @BeforeMethods("setUp")
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideArrays"
        * })
        */
        public function benchOuter($params): void
        {
            \NDArray::outer($this->testArray, $this->testArray);
        }

        public function provideArrays() {
            yield ['testArray' => \NDArray::ones([100])];
            yield ['testArray' => \NDArray::ones([500])];
            yield ['testArray' => \NDArray::ones([1000])];
        }
    }
?>
