<?php
    /**
     * @Groups({"initializers"})
     */
    class ArrayInitializerBench
    {
        /**
        * @var matrix
        */
        private $matrix = [];

        public function setUp(array $params): void
        {
            $this->matrix = $params['matrix'];
        }
        
        /**
        * @BeforeMethods("setUp")
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideMatrix",
        * })
        */
        public function benchArray($params)
        {
            $ndarray = \NDArray::array($this->matrix);
        }

        public function provideMatrix() {
            yield ['matrix' => \NDArray::zeros([1, 100])];
            yield ['matrix' => \NDArray::zeros([1, 500])];
            yield ['matrix' => \NDArray::zeros([1, 1000])];
            yield ['matrix' => \NDArray::zeros([10, 100])];
            yield ['matrix' => \NDArray::zeros([1000, 500])];
            yield ['matrix' => \NDArray::zeros([10000, 1000])];
        }
    }
?>
